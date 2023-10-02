from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, split_data_binary, split_data_multiclass, train_model_binary, train_model_multiclass


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data_binary,
                inputs=["model_input_table", "params:model_options_binary"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_binary_node",
            ),
            node(
                func=split_data_multiclass,
                inputs=["model_input_table", "params:model_options_multiclass"],
                outputs=["X_train_multi", "X_test_multi", "y_train_multi", "y_test_multi"],
                name="split_data_multiclass_node",
            ),
            node(
                func=train_model_binary,
                inputs=["X_train", "y_train"],
                outputs="binary_classifier",
                name="train_model_binary_node",
            ),
            node(
                func=train_model_multiclass,
                inputs=["X_train_multi", "y_train_multi"],
                outputs="multiclass_classifier",
                name="train_model_multiclass_node",
            ),
            node(
                func=evaluate_model,
                inputs=["binary_classifier", "X_test", "y_test"],
                name="evaluate_binary_model_node",
                outputs="metrics_binary",
            ),
            node(
                func=evaluate_model,
                inputs=["multiclass_classifier", "X_test_multi", "y_test_multi"],
                name="evaluate_multiclass_model_node",
                outputs="metrics_multiclass",
            ),
        ]
    )
