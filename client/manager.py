import asyncio
import hashlib
import json
import logging
from typing import List

import aioredis
import sentry_sdk
from opensearchpy import AsyncOpenSearch, helpers, OpenSearchException, OpenSearch

from dtype.client.body import PredictBody
from dtype.client.settings import Settings
from dtype.model.graph import Graph

SETTINGS_DEFAULT_PATH = "settings.json"


class Manager:
    """
    Class to manage loading, reading, writing, and logging of the service.

    Args:
        settings (Settings): Service secrets and parameters.
        redis_client (aioredis.Redis): Redis client for results caching.
        opensearch_client (AsyncOpenSearch): Opensearch client to store next iteration training data.

    Attributes:
        settings (Settings): Service secrets and parameters.
        redis_client (aioredis.Redis): Redis client for results caching.
        opensearch_client (AsyncOpenSearch): Opensearch client to store next iteration training data.
    """

    settings: Settings
    redis_client: aioredis.Redis
    opensearch_client: AsyncOpenSearch
    opensearch_sync_client: OpenSearch

    def __init__(
            self,
            settings: Settings,
            redis_client: aioredis.Redis,
            opensearch_client: AsyncOpenSearch,
            opensearch_sync_client: OpenSearch
    ) -> None:
        self._settings = settings
        """Settings: Protected settings property."""
        self.redis_client = redis_client
        """aioredis.Redis: Redis client for results caching."""
        self.opensearch_client = opensearch_client
        """AsyncOpenSearch: Opensearch client to store next iteration training data."""
        self.opensearch_sync_client = opensearch_sync_client
        """OpenSearch: Opensearch client to store next iteration training data."""

    @staticmethod
    def load_settings() -> Settings:
        """
            Load service secrets and parameters.
        Returns:
            Settings: loaded settings.
        """
        with open(SETTINGS_DEFAULT_PATH) as fp:
            settings = json.load(fp)
        return settings

    @property
    def settings(self) -> Settings:
        """Settings: settings property getter."""
        return self._settings

    @staticmethod
    def get_module_name(file_path: str) -> str:
        """
        Convert full file path to dot seperated module name.
        Args:
            file_path (str): full file path.

        Returns:
            str: Dot seperated module name.
        """
        return ".".join(file_path.split("/")[-2:])

    @classmethod
    def setup(cls) -> "Manager":
        """
        Set up the service.
        Returns:
            Manager: Created manager object to manage all external i/o ops and logging.
        """
        # (1) Loading settings
        with open(SETTINGS_DEFAULT_PATH) as fp:
            settings = Settings(**json.load(fp))

        # (2) Initializing logger
        logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.DEBUG)

        # (3) Setting up GlitchTip (Sentry)
        sentry_sdk.init(settings.sentry_dsn)

        # (4) Setup Redis client
        redis_client = aioredis.from_url(settings.redis_host, encoding="utf8", decode_responses=True)

        # (5) Setup Opensearch client
        opensearch_client = AsyncOpenSearch(
            hosts=[settings.opensearch_host],
            http_auth=(settings.opensearch_user, settings.opensearch_pass),
            verify_certs=settings.opensearch_verify_certs,
        )

        opensearch_sync_client = OpenSearch(
            hosts=[settings.opensearch_host],
            http_auth=(settings.opensearch_user, settings.opensearch_pass),
            verify_certs=settings.opensearch_verify_certs,
        )

        return cls(
            settings=settings,
            redis_client=redis_client,
            opensearch_client=opensearch_client,
            opensearch_sync_client=opensearch_sync_client
        )

    @classmethod
    def log(cls, message: str, level: int, file_path: str) -> None:
        """
        Log a message to service logs and sentry.
        Args:
            message (str): Log message.
            level (int): Log level, possible values are logging.INFO, logging.DEBUG, logging.WARNING, logging.ERROR and
                logging.CRITICAL.
            file_path (str):  File path to get module name from.
        """
        # (0) Initializing logger
        logger = logging.getLogger(file_path)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # (1) Getting custom logger with module name
        module_name = cls.get_module_name(file_path=file_path)

        # (2) Firing the right log level with the given message
        logger.log(level=level, msg=f"{module_name} - {message}")

    async def cache_result(self, id: str, graph: Graph, result: str) -> None:
        """
        Cache result to be rated later.
        Args:
            id (str): Graph result uuid.
            graph (Graph): Adjacency list based graph data structure.
            result (str): Predicted class for the input graph.
        """
        store_dict = {"input": graph.dict(), "output": result}

        await self.redis_client.set(name=id, value=json.dumps(store_dict))

    async def cache_results(self, ids: List[str], body: PredictBody, results: List[str]) -> None:
        """
        Cache result to be rated later.
        Args:
            ids (List[str]): Graph result uuids.
            body (PredictBody): Request body contains list of graphs.
            results (List[str]): Predicted class for the input graphs.
        """
        await asyncio.gather(
            *[self.cache_result(id=ids[i], graph=graph, result=results[i]) for i, graph in enumerate(body.graphs)]
        )

    async def get_from_cache(self, id: str) -> dict:
        """
        Get result from redis cache.
        Args:
            id (str): Result track uuid.

        Returns:
            dict: Cached result dict with "input" and "output fields.
        """
        result = await self.redis_client.get(name=id)
        if result is None:
            raise ValueError(f"Invalid rating id {id}")

        return json.loads(result)

    async def store_wrong_answers(self, answers: List[dict]) -> None:
        """
        Store rated results to be trained on later.
        Args:
            answers (List[dict]):
        """

        def _determine_action(answer: dict, operation: str, index: str) -> dict:
            """
            Get the action of Opensearch operation.
            Args:
                answer (dict): Answer dict includes the input and correct output.
                operation (str): Operation type.
                index (str): Index to store in.

            Returns:
                dict: Action of Opensearch operation.

            Raises:
                OpenSearchException: On failed insert/update operation.
            """
            hash_object = hashlib.sha256()
            hash_object.update(json.dumps(answer).encode("utf-8"))

            return {
                "_op_type": operation,
                "_index": index,
                "_id": hash_object.hexdigest(),
                "doc": answer,
                "doc_as_upsert": True,
            }

        try:
            await helpers.async_bulk(
                self.opensearch_client,
                actions=[_determine_action(answer, "update", self.settings.opensearch_index) for answer in answers],
                index=self.settings.opensearch_index,
                raise_on_error=True,
                request_timeout=60,
            )
        except Exception as e:
            error_msg = f"Opensearch bulk insert/update operation failed with the error {str(e)}"
            Manager.log(message=error_msg, level=logging.ERROR, file_path=__file__)

            raise OpenSearchException(error_msg)
