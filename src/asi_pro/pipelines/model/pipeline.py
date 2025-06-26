from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load_data, train_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=load_data,
            inputs="params:image_folder",
            outputs="dls",
            name="load_data_node"
        ),
        node(
            func=train_model,
            inputs=["dls", "params:model_output_path"],
            outputs=None,
            name="train_model_node"
        ),
    ])
