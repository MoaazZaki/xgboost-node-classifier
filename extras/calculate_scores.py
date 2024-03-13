import pandas as pd
from sklearn.metrics import classification_report


def extract_labels(original_df, extended_df):
    y_pred = []
    y_actual = []
    for graph_id, node_id, prediction in extended_df.values:
        graph_df = original_df[original_df['graphId'] == graph_id]
        label = graph_df[
            (graph_df['originNodeId'] == node_id) | (graph_df['destinationNodeId'] == node_id)
            ]['label'].unique().tolist()[0]
        y_pred.append(prediction)
        y_actual.append(label)

    return y_actual, y_pred


def print_report(y_actual, y_pred):
    print("== Classification Report ==")
    print(
        classification_report(y_actual, y_pred)
    )


if __name__ == '__main__':
    df_extended = pd.read_parquet('.cache/extended_validation.parquet', engine='fastparquet')
    df_original = pd.read_parquet('.cache/validation.parquet', engine='fastparquet')
    y_actual, y_pred = extract_labels(
        original_df=df_original,
        extended_df=df_extended
    )

    print_report(y_actual, y_pred)
