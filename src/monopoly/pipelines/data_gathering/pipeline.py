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
        ]
    )