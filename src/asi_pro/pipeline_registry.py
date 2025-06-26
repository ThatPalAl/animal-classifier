from kedro.pipeline import Pipeline
from asi_pro.pipelines.model import create_pipeline as model_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "__default__": model_pipeline(),
    }
