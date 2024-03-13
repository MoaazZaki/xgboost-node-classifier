import asyncio
import logging
from typing import List, Dict, Tuple, Any

import joblib
import networkx as nx
from xgboost import XGBClassifier

from client.manager import Manager
from dtype.model.graph import Graph

NO_INITIALIZED_MODEL_ERROR = (
    "Graph model XGBClassifier is not initialized and no path given,"
    " please class GraphXGB with providing path of the model."
)

INCORRECT_FEATURES_NUMBER_ERROR = (
    "Graph feature generation must generate {correct_features_num} features, but was given {generated_features_num}"
)

EIGENVECTOR_NO_CONVERGENCE_ERROR = "Eigenvector centrality Convergence failed for the following graph:\n {graph}"

TRAINING_CORRUPT_DATA_ERROR = (
    "Found graph with \"{unknown_token}\" label, please check if any of the \"output\" "
    "values given in the request are not of type \"{unknown_token}\""
)

EIGENVECTOR_centrality_MAX_ITER = 10000

MOTIF_TYPES = [
    "003",
    "012",
    "102",
    "021D",
    "021U",
    "021C",
    "111D",
    "111U",
    "030T",
    "030C",
    "201",
    "120D",
    "120U",
    "120C",
    "210",
    "300",
]


class GraphXGB:
    """
    Singleton implementation of Graph XGBoost model.

    Args:
        path (:obj:`str`, optional): Model path to load from. Defaults to None.
        num_features (:obj:`int`, optional): Vector dimension per graph node. Defaults to 107.
        num_classes (:obj:`int`, optional): Number of possible nodes classes/types. Defaults to 40.
        unknown_token (:obj:`str`, optional)): Unknown node class label. Defaults to "UNK".

    Attributes:
        __instance__ (:obj:`GraphXGB`, optional): Singleton instance of the class. Defaults to None.
        __model__ (:obj:`XGBClassifier`, optional): XGBClassifier model. Defaults to None.
        __num_classes__ (:obj:`int`, optional): Number of possible nodes classes/types. Defaults to 40.
        __num_features__ (:obj:`int`, optional): Vector dimension per graph node. Defaults to 107.
        __classes__ (:obj:`List[str]`, optional): Classes possible values list. Defaults to [].
        __unknown_token__ (:obj:`str`, optional): Unknown node class label. Defaults to "UNK".
    """

    __instance__: "GraphXGB" = None
    __model__: XGBClassifier = None
    __num_classes__: int = 40
    __num_features__: int = 107
    __classes__: List[str] = []
    __unknown_token__: str = "UNK"

    def __new__(
            cls, path: str = None, num_features: int = 107, num_classes: int = 40, unknown_token: str = "UNK"
    ) -> "GraphXGB":
        if cls.__instance__ is None:
            if path is None:
                Manager.log(message=NO_INITIALIZED_MODEL_ERROR, level=logging.ERROR, file_path=__file__)
                raise RuntimeError(NO_INITIALIZED_MODEL_ERROR)
            cls.__instance__: "GraphXGB" = super(GraphXGB, cls).__new__(cls)
            cls.__model__: XGBClassifier = joblib.load(filename=path)
            cls.__num_classes__: int = num_classes
            cls.__num_features__: int = num_features
            cls.__classes__: List[str] = [f"N{i}" for i in range(num_classes)]
            cls.__unknown_token__: str = unknown_token

        return cls.__instance__

    @staticmethod
    def construct_nx_graph(graph: Graph) -> nx.DiGraph:
        """
        Greate nx.DiGraph object from json adjacency list based graph.
        Args:
            graph (Graph): Adjacency list based graph.

        Returns:
            nx.DiGraph: Constructed graph.
        """
        nx_graph = nx.DiGraph()

        for edge in graph.edges:
            nx_graph.add_edge(edge.origin_id, edge.destination_id)
            nx_graph.nodes[edge.origin_id]["type"] = edge.origin_type
            nx_graph.nodes[edge.destination_id]["type"] = edge.destination_type

        return nx_graph

    def get_neighbors_map(self, nx_graph: nx.DiGraph, node_id_to_type: Dict[str, str]) -> Dict[str, List[int]]:
        """
        Get map which holds each node as key, and its neighbor types count.
        Args:
            nx_graph (nx.DiGraph): Input graph.
            node_id_to_type (Dict[str, str]): Node ID to type map.

        Returns:
            Dict[str, List[int]]: Dict with node ID as key, and vector of counts with dimension equals to num_classes.
        """
        node_neighbors = {node: list(nx_graph.neighbors(node)) for node in nx_graph.nodes()}
        for node_id, neighbors in node_neighbors.items():
            neighbors_types = [node_id_to_type[neighbor] for neighbor in neighbors]
            neighbors_vector = [neighbors_types.count(node_type) for node_type in self.__classes__]
            node_neighbors[node_id] = neighbors_vector

        return node_neighbors

    def get_shortest_paths_map(self, nx_graph: nx.DiGraph, node_id_to_type: Dict[str, str]) -> Dict[str, List[int]]:
        """
        Get map which holds each node as key, and all shortest paths to each type.
        Args:
            nx_graph (nx.DiGraph): Input graph.
            node_id_to_type (Dict[str, str]): Node ID to type map.

        Returns:
            Dict[str, List[int]]: Dict with node ID as key, and vector of shortest paths with dimension equals
                to num_classes.
        """
        node_path_features = {
            node_type: node_path for node_type, node_path in nx.all_pairs_shortest_path_length(nx_graph)
        }
        for node_id, _paths in node_path_features.items():
            paths_vector = [0] * self.__num_classes__
            for path_node_id, steps in _paths.items():
                if node_id_to_type[path_node_id] != self.__unknown_token__:
                    paths_vector[int(node_id_to_type[path_node_id][1:])] = steps

            node_path_features[node_id] = paths_vector

        return node_path_features

    def get_unknown_node_vector(self, **kwargs) -> List[float]:
        """
        Get the vector representation of the unknown node in the graph.
        Args:
            **kwargs: Graph features.

        Returns:
            List[float]: Feature vectors.
        """
        for i, node_id in enumerate(kwargs.get("nodes_id")):
            if kwargs.get("node_id_to_type").get(node_id) == self.__unknown_token__:
                features = [
                    kwargs.get("in_degree").get(node_id),
                    kwargs.get("out_degree").get(node_id),
                    kwargs.get("degree").get(node_id),
                    kwargs.get("betweenness_centrality").get(node_id),
                    kwargs.get("eigenvector_centrality").get(node_id),
                    kwargs.get("graph_density"),
                    kwargs.get("pagerank_centrality").get(node_id),
                    kwargs.get("katz_centrality").get(node_id),
                    kwargs.get("degree_centrality").get(node_id),
                    kwargs.get("closeness_centrality").get(node_id),
                    kwargs.get("local_clustering_coefficient").get(node_id)
                ]

                features.extend(kwargs.get("nodes_neighbors", {}).get(node_id))
                features.extend(kwargs.get("nodes_path_features", {}).get(node_id))
                features.extend(kwargs.get("graph_motifs"))

                if len(features) != self.__num_features__:
                    error_msg = INCORRECT_FEATURES_NUMBER_ERROR.format(
                        correct_features_num=self.__num_features__, generated_features_num=len(features)
                    )
                    Manager.log(message=error_msg, level=logging.FATAL, file_path=__file__)

                    raise RuntimeError(error_msg)

                return features

    def get_labelled_graph(self, **kwargs) -> Tuple[List[List[float]], List[int]]:
        """
        Get the vector representation of whole graph nodes.
        Args:
            **kwargs: Graph features.

        Returns:
            Tuple[List[List[float]], List[int]]: Graph features and labels.
        """
        graph_vectors = []
        graph_labels = []

        node_id_to_type = kwargs.get("node_id_to_type")
        for i, node_id in enumerate(kwargs.get("nodes_id")):
            features = [
                kwargs.get("in_degree").get(node_id),
                kwargs.get("out_degree").get(node_id),
                kwargs.get("degree").get(node_id),
                kwargs.get("betweenness_centrality").get(node_id),
                kwargs.get("eigenvector_centrality").get(node_id),
                kwargs.get("graph_density"),
                kwargs.get("pagerank_centrality").get(node_id),
                kwargs.get("katz_centrality").get(node_id),
                kwargs.get("degree_centrality").get(node_id),
                kwargs.get("closeness_centrality").get(node_id),
                kwargs.get("local_clustering_coefficient").get(node_id)
            ]

            features.extend(kwargs.get("nodes_neighbors", {}).get(node_id))
            features.extend(kwargs.get("nodes_path_features", {}).get(node_id))
            features.extend(kwargs.get("graph_motifs"))

            if len(features) != self.__num_features__:
                error_msg = INCORRECT_FEATURES_NUMBER_ERROR.format(
                    correct_features_num=self.__num_features__, generated_features_num=len(features)
                )
                Manager.log(message=error_msg, level=logging.FATAL, file_path=__file__)

                raise RuntimeError(error_msg)

            node_type = node_id_to_type.get(node_id)
            if node_type == self.__unknown_token__:
                error_msg = TRAINING_CORRUPT_DATA_ERROR.format(
                    unknown_token=self.__unknown_token__
                )
                Manager.log(message=error_msg, level=logging.ERROR, file_path=__file__)

                raise ValueError(error_msg)

            graph_vectors.append(features)
            graph_labels.append(int(node_type[1:]))
        return graph_vectors, graph_labels

    async def get_features(self, graph: Graph) -> List[float]:
        """
        Get adjacency list based unknown node vector representation.
        Args:
            graph (Graph): Input graph.

        Returns:
            List[float]: Unknown node vector representation
        """
        nx_graph = self.construct_nx_graph(graph=graph)
        node_id_to_type = nx.get_node_attributes(nx_graph, "type")

        # (1) Node level features
        # (1.1) Scalar per node features

        in_degree = dict(nx_graph.in_degree())
        out_degree = dict(nx_graph.out_degree())
        degree = dict(nx_graph.degree())

        degree_centrality = nx.degree_centrality(nx_graph)
        betweenness_centrality = nx.betweenness_centrality(nx_graph)
        katz_centrality = nx.katz_centrality(nx_graph)
        closeness_centrality = nx.closeness_centrality(nx_graph)
        pagerank_centrality = nx.pagerank(nx_graph)
        try:
            eigenvector_centrality = nx.eigenvector_centrality(nx_graph, max_iter=EIGENVECTOR_centrality_MAX_ITER)
        except nx.PowerIterationFailedConvergence:
            eigenvector_centrality = {}
            Manager.log(
                message=EIGENVECTOR_NO_CONVERGENCE_ERROR.format(graph=graph.json()),
                level=logging.WARNING,
                file_path=__file__,
            )

        local_clustering_coefficient = nx.clustering(nx_graph)

        # (1.2) Vector per node features
        nodes_neighbors = self.get_neighbors_map(nx_graph=nx_graph, node_id_to_type=node_id_to_type)
        nodes_path_features = self.get_shortest_paths_map(nx_graph=nx_graph, node_id_to_type=node_id_to_type)

        # (2) Graph level features
        # (2.1) Scalar graph features
        graph_density = nx.density(nx_graph)

        # (2.2) Vector graph features
        graph_motifs = nx.triadic_census(nx_graph)
        graph_motifs = [graph_motifs[subgraph] for subgraph in MOTIF_TYPES]

        kwargs = dict(
            nodes_id=nx_graph.nodes(),
            node_id_to_type=node_id_to_type,
            in_degree=in_degree,
            out_degree=out_degree,
            degree=degree,
            degree_centrality=degree_centrality,
            betweenness_centrality=betweenness_centrality,
            katz_centrality=katz_centrality,
            closeness_centrality=closeness_centrality,
            pagerank_centrality=pagerank_centrality,
            eigenvector_centrality=eigenvector_centrality,
            local_clustering_coefficient=local_clustering_coefficient,
            graph_density=graph_density,
            nodes_neighbors=nodes_neighbors,
            nodes_path_features=nodes_path_features,
            graph_motifs=graph_motifs,
        )

        return self.get_unknown_node_vector(**kwargs)

    def get_labelled_features(self, graph: Graph):
        """
        Get adjacency list based all nodes vector representation.
        Args:
            graph (Graph): Input graph.

        Returns:
            Tuple[List[List[float]], List[int]]: All nodes vector representation and labels
        """
        nx_graph = self.construct_nx_graph(graph=graph)
        node_id_to_type = nx.get_node_attributes(nx_graph, "type")

        # (1) Node level features
        # (1.1) Scalar per node features

        in_degree = dict(nx_graph.in_degree())
        out_degree = dict(nx_graph.out_degree())
        degree = dict(nx_graph.degree())

        degree_centrality = nx.degree_centrality(nx_graph)
        betweenness_centrality = nx.betweenness_centrality(nx_graph)
        katz_centrality = nx.katz_centrality(nx_graph)
        closeness_centrality = nx.closeness_centrality(nx_graph)
        pagerank_centrality = nx.pagerank(nx_graph)
        try:
            eigenvector_centrality = nx.eigenvector_centrality(nx_graph, max_iter=EIGENVECTOR_centrality_MAX_ITER)
        except nx.PowerIterationFailedConvergence:
            eigenvector_centrality = {}
            Manager.log(
                message=EIGENVECTOR_NO_CONVERGENCE_ERROR.format(graph=graph.json()),
                level=logging.WARNING,
                file_path=__file__,
            )

        local_clustering_coefficient = nx.clustering(nx_graph)

        # (1.2) Vector per node features
        nodes_neighbors = self.get_neighbors_map(nx_graph=nx_graph, node_id_to_type=node_id_to_type)
        nodes_path_features = self.get_shortest_paths_map(nx_graph=nx_graph, node_id_to_type=node_id_to_type)

        # (2) Graph level features
        # (2.1) Scalar graph features
        graph_density = nx.density(nx_graph)

        # (2.2) Vector graph features
        graph_motifs = nx.triadic_census(nx_graph)
        graph_motifs = [graph_motifs[subgraph] for subgraph in MOTIF_TYPES]

        kwargs = dict(
            nodes_id=nx_graph.nodes(),
            node_id_to_type=node_id_to_type,
            in_degree=in_degree,
            out_degree=out_degree,
            degree=degree,
            degree_centrality=degree_centrality,
            betweenness_centrality=betweenness_centrality,
            katz_centrality=katz_centrality,
            closeness_centrality=closeness_centrality,
            pagerank_centrality=pagerank_centrality,
            eigenvector_centrality=eigenvector_centrality,
            local_clustering_coefficient=local_clustering_coefficient,
            graph_density=graph_density,
            nodes_neighbors=nodes_neighbors,
            nodes_path_features=nodes_path_features,
            graph_motifs=graph_motifs,
        )

        return self.get_labelled_graph(**kwargs)

    async def predict(self, graph: Graph) -> str:
        """
        Predict the missing node label.
        Args:
            graph (Graph): Adjacency list based graph.

        Returns:
            str: Class label.
        """
        features = await self.get_features(graph=graph)
        return f"N{self.__model__.predict(X=[features]).tolist()[0]}"

    async def __call__(self, graphs: List[Graph]) -> Tuple[str]:
        """
        Batch predict missing nodes for input graphs.
        Args:
            graphs (List[Graph]): Input graphs.

        Returns:
            Tuple[str]: predicted labels.
        """
        return await asyncio.gather(*[self.predict(graph=graph) for graph in graphs])
