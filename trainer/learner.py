import asyncio
import os
from datetime import datetime
from typing import List, Tuple

import joblib
import numpy as np

from client.manager import Manager
from dtype.model.graph import Graph
from model.xgboost import GraphXGB


class Learner:
    """
    Class responsible to initiate learning step for the training.

    Args:
        manager (Manager): Training manager object.

    Attributes:
        model_path (str): Model path to load/save.
    """
    model_path: str

    def __init__(self, manager: Manager) -> None:
        self.model = GraphXGB(
            path=manager.settings.model_path,
            num_features=manager.settings.num_features,
            num_classes=manager.settings.num_classes,
            unknown_token=manager.settings.unknown_token
        )

        self.model_path = manager.settings.model_path

    def prepare_single_graph(self, graph_json: dict) -> np.ndarray:
        """
        Extract features and labels from JSON graph
        Args:
            graph_json (dict): Ajacency list based graph in JSON format.

        Returns:
            np.ndarray: Graph features and labels.
        """
        graph_json['input']['edges'] = [
            {
                "origin_id": edge['origin_id'],
                "origin_type": (
                    edge["origin_type"]
                    if edge["origin_type"] != self.model.__unknown_token__
                    else graph_json["output"]
                ),
                "destination_id": edge["destination_id"],
                "destination_type": (
                    edge["destination_type"]
                    if edge["destination_type"] != self.model.__unknown_token__
                    else graph_json["output"]
                )
            }
            for edge in graph_json['input']['edges']
        ]

        graph = Graph(
            **graph_json['input']
        )

        features, labels = self.model.get_labelled_features(graph=graph)

        return np.column_stack([features, labels])

    def prepare(self, raw_data: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and labels from JSON graphs
        Args:
            raw_data (List[dict]): List of JSON format graphs.

        Returns:
            Tuple[np.ndarray, np.ndarray]: X & y to train on.
        """
        processed_data = np.vstack(
            [
                self.prepare_single_graph(graph_json=graph_json["_source"])
                for graph_json in raw_data
            ]
        )

        features, labels = processed_data[:, :-1], processed_data[:, -1]
        return features, labels

    def start(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Start learning by fitting an ew model.
        Args:
            features (np.ndarray): X matrix to fit the model on (shape:num_nodes,num_classes).
            labels (np.ndarray): X matrix to fit the model on (shape:num_nodes).
        """
        self.model.__model__ = self.model.__model__.fit(
            X=features,
            y=labels
        )

    def save(self, manager: Manager) -> None:
        """
        Save the model.
        Args:
            manager (Manager): Trainer manager.
        """
        os.remove(self.model_path)
        joblib.dump(self.model.__model__, self.model_path)

        asyncio.get_event_loop().run_until_complete(
            manager.opensearch_client.index(
                index=manager.settings.trainer_status_index,
                body={
                    "created_timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')
                }
            )
        )

    def clean(self) -> None:
        """
        Clean the current model instance till the next run.
        """
        del self.model.__model__
