"""
This file is just a quick script to insert data into opensearch to test training.
"""

import hashlib
import json
from datetime import datetime

from tqdm import tqdm
import pandas as pd
from opensearchpy import OpenSearch, helpers

DATA = ".cache/random.parquet"

opensearch = OpenSearch(
    ['https://localhost:9200'],
    http_auth=('admin', 'admin'),
    verify_certs=False  # Set to True if your OpenSearch server has a valid SSL certificate
)


def insert_data(df: pd.DataFrame):
    graphs = []

    for graph_id, graph_data in tqdm(df.groupby("graphId")):
        edges = []
        node_type = None
        for source_node_id, source_node_type, dest_node_id, dest_node_type, label in zip(
                graph_data["originNodeId"].tolist(),
                graph_data["originNodeType"].tolist(),
                graph_data["destinationNodeId"].tolist(),
                graph_data["destinationNodeType"].tolist(),
                graph_data["label"].tolist()
        ):

            edges.append(
                {
                    'origin_id': source_node_id,
                    'origin_type': source_node_type,
                    'destination_id': dest_node_id,
                    'destination_type': dest_node_type
                }
            )

            if (source_node_type == 'UNK' or dest_node_type == 'UNK') and label is not None:
                node_type = label

        if node_type is None:
            raise Exception('Unknown Node type is missing in the given data!')

        graphs.append(
            {
                "input": {
                    "edges": edges
                },
                "output": node_type,
                "added_timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')
            }
        )

    def _determine_action(answer: dict, operation: str, index: str) -> dict:
        hash_object = hashlib.sha256()
        hash_object.update(json.dumps(answer).encode("utf-8"))

        return {
            "_op_type": operation,
            "_index": index,
            "_id": hash_object.hexdigest(),
            "doc": answer,
            "doc_as_upsert": True,
        }

    opensearch_actions = [
        _determine_action(answer, "update", "graphs_data")
        for answer in graphs
    ]
    helpers.bulk(
        opensearch,
        actions=opensearch_actions,
        index="graphs_data",
        raise_on_error=True,
        request_timeout=60,
    )


if __name__ == '__main__':
    val_df = pd.read_parquet(DATA, engine='fastparquet')
    insert_data(df=val_df)
