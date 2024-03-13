import asyncio
import logging

from client.manager import Manager

DEFAULTS_STATUS_INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "created_timestamp": {
                "type": "date",
                "format": "strict_date_optional_time||epoch_millis"
            },
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    }
}


class Loader:
    """
    Class to load data for the trainer.
    """
    @staticmethod
    def should_start(manager: Manager) -> bool:
        """
        Function to determine if training should start or not, by checking if we have enough data.
        Args:
            manager (Manager): Trainer manager.

        Returns:
            bool: Flag indicates if the training should start.
        """
        data_index_exists = manager.opensearch_sync_client.indices.exists(index=manager.settings.opensearch_index)

        if not data_index_exists:
            return False

        status_index_exists = manager.opensearch_sync_client.indices.exists(index=manager.settings.trainer_status_index)
        last_known_trained_size = 0

        if status_index_exists:
            status_index_count = manager.opensearch_sync_client.count(
                index=manager.settings.trainer_status_index
            )["count"]

            if status_index_count != 0:
                last_entry_query = {
                    "size": 1,
                    "sort": [
                        {
                            "created_timestamp": {
                                "order": "desc"
                            }
                        }
                    ]
                }

                last_training_timestamp = manager.opensearch_sync_client.search(
                    index=manager.settings.trainer_status_index,
                    body=last_entry_query
                )['hits']['hits'][0]['_source']['created_timestamp']

                less_than_last_timestamp_query: dict = {
                    "query": {
                        "range": {
                            "added_timestamp": {
                                "gt": last_training_timestamp
                            }
                        }
                    }
                }

                last_known_trained_size = manager.opensearch_sync_client.count(
                    body=less_than_last_timestamp_query,
                    index=manager.settings.opensearch_index
                )["count"]
        else:
            manager.opensearch_sync_client.indices.create(
                index=manager.settings.trainer_status_index,
                body=DEFAULTS_STATUS_INDEX_MAPPING
            )

        current_train_size = manager.opensearch_sync_client.count(
            body={
                "query": {
                    "match_all": {}
                }
            },
            index=manager.settings.opensearch_index
        )["count"]

        added_train_size = current_train_size - last_known_trained_size

        return added_train_size >= manager.settings.trainer_minimum_dataset_size

    @staticmethod
    def get(manager: Manager) -> list:
        """
        Get training data.
        Args:
            manager (Manager): Trainer manager.

        Returns:
            list: Graphs to train on.
        """
        body = {
            "query": {
                "match_all": {}
            },
            "size": 10000
        }

        page = manager.opensearch_sync_client.search(
            index=manager.settings.opensearch_index,
            body=body,
            scroll='2m'
        )

        scroll_id = page['_scroll_id']
        scroll_size = page['hits']['total']['value']

        all_docs = []
        while len(all_docs) < scroll_size:
            all_docs.extend(page['hits']['hits'])
            page = manager.opensearch_sync_client.scroll(scroll_id=scroll_id, scroll='2m')
            scroll_id = page['_scroll_id']

        return all_docs
