import mlflow
from pathlib import Path


def setup_mlflow(experiment_name: str = "quantvol-volatility-forecasting"):
    """Configure MLflow to use local file-based tracking and set the experiment."""
    tracking_dir = Path("mlruns").absolute()
    mlflow.set_tracking_uri(f"file:///{tracking_dir}")
    mlflow.set_experiment(experiment_name)

def log_run(model_name: str, params: dict, metrics: dict, tags: dict = None):
    """
    Log a single model run to MLflow: hyperparameters, evaluation metrics,
    and optional tags (e.g., notes about the experiment).
    """
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        if tags:
            mlflow.set_tags(tags)

        mlflow.set_tag("model_type", model_name)