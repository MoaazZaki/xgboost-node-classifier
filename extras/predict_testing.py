"""
This file is just a quick script to produce labels for validation.parquet & testing.parquet
"""
import os

import pandas as pd
import requests

from tqdm import tqdm

datasets_to_predict = [".cache/validation.parquet", ".cache/test.parquet"]


def send_request(request_body):
    headers = {
        "Api-key": "3Fb6KpWx8RjDq2YvHs7eLt9gXuZoNfArEiCbPmGy1SdQt5UoIwAxQzVcJnMkOzLp"
    }
    response = requests.post('http://127.0.0.1:80/predict', headers=headers, json=request_body)
    response.raise_for_status()
    return response.json()['results'][0]


def produce_labels(df):
    graphs = []

    for graph_id, graph_data in tqdm(df.groupby("graphId")):
        edges = []
        unknown_node_id = None
        for source_node_id, source_node_type, dest_node_id, dest_node_type in zip(
                graph_data["originNodeId"].tolist(),
                graph_data["originNodeType"].tolist(),
                graph_data["destinationNodeId"].tolist(),
                graph_data["destinationNodeType"].tolist()
        ):

            edges.append(
                {
                    'origin_id': source_node_id,
                    'origin_type': source_node_type,
                    'destination_id': dest_node_id,
                    'destination_type': dest_node_type
                }
            )

            if unknown_node_id is None:
                if source_node_type == 'UNK':
                    unknown_node_id = source_node_id
                elif dest_node_type == 'UNK':
                    unknown_node_id = dest_node_id

        if unknown_node_id is None:
            raise Exception('Unknown Node type is missing in the given data!')

        label = send_request(
            request_body={
                "graphs": [
                    {
                        "edges": edges
                    }
                ]
            }
        )

        graphs.append(
            {
                "graphId": graph_id,
                "nodeId": unknown_node_id,
                "prediction": label
            }
        )
    return graphs


if __name__ == '__main__':
    for dataset in datasets_to_predict:
        df = pd.read_parquet(dataset, engine='fastparquet')
        data_name = os.path.basename(dataset)
        pd.DataFrame(produce_labels(df=df)).to_parquet(
            os.path.join(dataset.replace(data_name, ""), f"extended_{data_name}")
        )
