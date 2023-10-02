from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_data,
                inputs=["iot_data"],
                outputs=["model_input_table","metrics_data"],
                name="create_model_input_table_node",
            ),
        ]
    )
