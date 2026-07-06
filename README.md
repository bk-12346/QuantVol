# QuantVol — Volatility Forecasting & Regime Research Platform

An end-to-end, production-style research system for forecasting S&P 500 realized volatility. Compares classical econometric models (HAR-RV, GARCH) against modern ML approaches (LightGBM, XGBoost), with leakage-safe walk-forward validation, SHAP-based interpretability, full MLflow experiment tracking, and a live, containerized FastAPI + Streamlit deployment.

## Key Results

| Model | RMSE | QLIKE | Notes |
|---|---|---|---|
| **HAR-RV** | **0.0119** | **0.0116** | Best performer — direct, well-specified econometric baseline |
| XGBoost | 0.0477 | 0.0457 | Comparable to LightGBM; richer feature set, no edge over HAR-RV |
| LightGBM | 0.0520 | 0.0437 | Comparable to XGBoost |
| GARCH(1,1) | 0.0646 | 0.2313 | Weakest — accuracy degrades over multi-step-ahead forecast horizon |

**Finding:** HAR-RV substantially outperforms both tree-based ML models, consistent with academic literature — realized volatility's strong autocorrelation is already well-captured by HAR-RV's daily/weekly/monthly structure, leaving limited room for more complex models to add value on this dataset. Optuna hyperparameter tuning for LightGBM was tested but did *not* generalize across the full walk-forward period (documented as a limitation of static train/val tuning for non-stationary time series — see `notebooks/` and MLflow logs).

## Why This Project

This project treats volatility forecasting as a real research problem:

- **No lookahead bias** — every feature is validated to use only information available at the time of prediction (see `tests/test_features.py`)
- **Proper backtesting** — walk-forward (expanding window) validation, not random splits
- **Literature-standard evaluation** — QLIKE loss (not just RMSE), the metric actually used in volatility forecasting research
- **Multiple volatility estimators** — close-to-close, Parkinson, and Garman-Klass, cross-validated against each other
- **Full interpretability** — SHAP analysis explaining what drives each prediction
- **Real deployment** — a live API and demo UI, not just a notebook

## Architecture

```
Data ingestion (yfinance) → Cleaning → Feature engineering (vol estimators, regime tags)
↓
Baseline models (HAR-RV, GARCH) + ML models (LightGBM, XGBoost)
↓
Walk-forward validation → QLIKE/RMSE evaluation → MLflow tracking
↓
SHAP interpretability → FastAPI deployment → Streamlit demo
```

Fully reproducible via DVC (`dvc repro`) from raw data to final feature set.

## Methodology Highlights

- **Data**: S&P 500 (`^GSPC`) daily OHLCV, 2015–present, via `yfinance`
- **Volatility estimators**: realized (close-to-close), Parkinson, and Garman-Klass — all validated against known literature relationships (close-to-close systematically higher than range-based estimators)
- **Regime tagging**: 2018 Q4 selloff, 2020 COVID crash, 2022 rate hikes — used for evaluation segmentation, deliberately excluded as a model *feature* to avoid look-ahead bias from hindsight-labeled regimes
- **Validation**: expanding-window walk-forward, retrained every 21 trading days, identical splits across all four models for a fair comparison
- **Leakage testing**: explicit unit tests confirming rolling-window calculations and regime-boundary logic don't leak future information
- **Interpretability**: SHAP analysis on XGBoost confirms `rv_lag_1d` (yesterday's realized vol) as the dominant predictive signal — consistent with volatility clustering

## Tech Stack

Python · pandas · scikit-learn · LightGBM · XGBoost · `arch` (GARCH) · statsmodels · SHAP · MLflow · Optuna · DVC · FastAPI · Streamlit · Docker · uv · pytest

## Running It

**Setup:**
```bash
uv sync
```

**Reproduce the full data pipeline:**
```bash
uv run dvc repro
```

**Run the full model comparison:**
```bash
uv run python -m quantvol.models.baselines.har_rv_model
uv run python -m quantvol.models.baselines.garch_model
uv run python -m quantvol.models.ml.lgbm_model
uv run python -m quantvol.models.ml.xgboost_model
uv run python -m quantvol.validation.metrics
```

**View experiment tracking dashboard:**
```bash
uv run mlflow ui
```

**Run tests:**
```bash
uv run pytest tests/ -v
```

**Run the full app (API + demo UI) via Docker:**
```bash
docker compose up --build
```
- API + interactive docs: `http://localhost:8000/docs`
- Streamlit demo: `http://localhost:8501`

## Project Structure

```
quantvol/
├── configs/               # Data, feature, and model configs (YAML)
├── src/quantvol/
│   ├── data/               # Ingestion, cleaning, volatility estimators
│   ├── features/           # ML feature engineering, regime tagging
│   ├── models/
│   │   ├── baselines/      # HAR-RV, GARCH
│   │   └── ml/             # LightGBM, XGBoost
│   ├── validation/         # Walk-forward harness, QLIKE/RMSE metrics
│   ├── tuning/             # Optuna hyperparameter search
│   ├── tracking/           # MLflow logging utilities
│   ├── api/                # FastAPI app, inference, schemas
│   └── app/                # Streamlit demo
├── notebooks/              # EDA and SHAP analysis
├── tests/                  # Leakage-safety and boundary tests
├── scripts/                # Model training, MLflow logging scripts
├── dvc.yaml                # Reproducible pipeline definition
├── Dockerfile / docker-compose.yml
└── pyproject.toml
```
## Key Findings

- **Volatility clustering and fat tails are clearly present** in S&P 500 returns (excess kurtosis of 15.8, negative skew of -0.65), motivating the use of GARCH-family and autocorrelation-aware models over naive constant-variance assumptions
- **HAR-RV's simplicity is a strength, not a limitation** — its direct modeling of realized volatility outperforms more flexible tree-based models on this dataset
- **GARCH's multi-step-ahead forecasts degrade with horizon**, converging toward long-run unconditional variance — a known and explicitly documented limitation, rather than a hidden weakness
- **Static hyperparameter tuning doesn't generalize well across regimes** — Optuna-tuned LightGBM performed *worse* in full walk-forward evaluation than untuned defaults, highlighting the importance of evaluating tuning choices across the full time period, not a single validation split

## Future Work

- CI/CD pipeline (GitHub Actions: lint, test, and build checks on push)
- Regime-segmented performance breakdown (does GARCH do relatively better during crises despite losing overall?)
- Walk-forward-aware hyperparameter tuning (nested time-series cross-validation)
- Extension to a multi-asset universe with cross-sectional features

