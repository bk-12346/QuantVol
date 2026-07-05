import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import mean_squared_error

from quantvol.models.ml.lgbm_model import FEATURE_COLS, TARGET_COL


def chronological_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    """
    Split data chronologically into train / validation / test.
    No shuffling - order is preserved, since this is time-series data.
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df

def objective(trial, train_df: pd.DataFrame, val_df: pd.DataFrame) -> float:
    """
    Optuna objective: suggests hyperparameters, trains on train_df,
    scores on val_df using RMSE. Optuna will try to minimize this.
    """
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbose": -1,
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "num_leaves": trial.suggest_int("num_leaves", 7, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(train_df[FEATURE_COLS], train_df[TARGET_COL])

    preds = model.predict(val_df[FEATURE_COLS])
    rmse = np.sqrt(mean_squared_error(val_df[TARGET_COL], preds))

    return rmse

def main(n_trials: int = 50):
    df = pd.read_csv("data/processed/ml_features.csv", parse_dates=["Date"])
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)

    train_df, val_df, test_df = chronological_split(df)

    print(f"Train: {len(train_df)} rows | Val: {len(val_df)} rows | Test: {len(test_df)} rows")

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_df, val_df), n_trials=n_trials)

    print("\nBest RMSE on validation set:", study.best_value)
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save best params for reuse in the walk-forward pipeline
    import json
    output_path = Path("configs/model_configs/lightgbm_tuned.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nSaved best params to {output_path}")


if __name__ == "__main__":
    main()