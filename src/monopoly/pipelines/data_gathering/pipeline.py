from kedro.pipeline import Pipeline, pipeline, node
from .nodes import *

def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=extract_transform_data,
                inputs=None,
                outputs="data",
                name="data_node",
            ),
            node(
                func=data_merge,
                inputs="data",
                outputs=["client_data", "client_transactions", "merged_data"],
                name="data_merge_node",
            ),
            node(
                func=data_groups,
                inputs="merged_data",
                outputs="data_groups",
                name="data_groups_node",
            ),
        ]
    )