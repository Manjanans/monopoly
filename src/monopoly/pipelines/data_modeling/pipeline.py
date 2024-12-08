from kedro.pipeline import Pipeline, pipeline, node
from .nodes import *

def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=clean_data,
                inputs=["client_data","params:quantile_international_save_path","params:quantile_num_acc_save_path","params:quantile_predict_international","params:quantile_predict_num_acc"],
                outputs="clean_data",
                name="clean",
            ),
            node(
                func=international_model,
                inputs=["clean_data","params:save_path_international_model"],
                outputs=None,
                name="international_model",
            ),
            node(
                func=num_acc_model,
                inputs=["clean_data","params:save_path_num_acc_model"],
                outputs=None,
                name="num_acc_model",
            ),
        ]
    )