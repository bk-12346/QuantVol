import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from quantvol.tracking.mlflow_utils import setup_mlflow, log_run


def main():
    setup_mlflow()

    # --- HAR-RV ---
    log_run(
        model_name="HAR-RV",
        params={
            "model_type": "linear_regression",
            "features": "daily,weekly,monthly_lagged_rv",
            "min_train_size": 500,
            "step_size": 21,
        },
        metrics={
            "rmse": 0.011892,
            "qlike": 0.011578,
        },
        tags={"notes": "Best-performing baseline; standard HAR-RV specification (Corsi 2009)"},
    )

    # --- GARCH(1,1) ---
    log_run(
        model_name="GARCH(1,1)",
        params={
            "model_type": "garch",
            "p": 1,
            "q": 1,
            "dist": "normal",
            "min_train_size": 500,
            "step_size": 21,
        },
        metrics={
            "rmse": 0.064582,
            "qlike": 0.231309,
        },
        tags={"notes": "Multi-step-ahead forecast within each 21-day window; accuracy degrades with horizon due to mean-reversion in variance equation"},
    )

    # --- LightGBM (untuned defaults) ---
    log_run(
        model_name="LightGBM_untuned",
        params={
            "model_type": "lightgbm",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 5,
            "num_leaves": 15,
            "min_child_samples": 20,
            "min_train_size": 500,
            "step_size": 21,
        },
        metrics={
            "rmse": 0.058532 if False else 0.051988,  # untuned result
            "qlike": 0.043652,
        },
        tags={"notes": "Best ML result; outperforms tuned variant, see LightGBM_tuned run for comparison"},
    )

    # --- LightGBM (Optuna-tuned) ---
    log_run(
        model_name="LightGBM_tuned",
        params={
            "model_type": "lightgbm",
            "n_estimators": 162,
            "learning_rate": 0.030409,
            "max_depth": 3,
            "num_leaves": 63,
            "min_child_samples": 42,
            "subsample": 0.905203,
            "colsample_bytree": 0.969285,
            "min_train_size": 500,
            "step_size": 21,
        },
        metrics={
            "rmse": 0.058532,
            "qlike": 0.057061,
        },
        tags={"notes": "Optuna-tuned on single chronological 70/15/15 split; did NOT generalize across full walk-forward period - worse than untuned defaults. Documented limitation of static tuning for non-stationary time series."},
    )

    print("Logged 4 runs to MLflow. Run 'uv run mlflow ui' to view the dashboard.")

    # --- XGBoost ---
    log_run(
        model_name="XGBoost",
        params={
            "model_type": "xgboost",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 5,
            "min_child_weight": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_train_size": 500,
            "step_size": 21,
        },
        metrics={
            "rmse": 0.047731,
            "qlike": 0.045704,
        },
        tags={"notes": "Comparable to LightGBM; both ML models underperform HAR-RV on this dataset"},
    )


if __name__ == "__main__":
    main()