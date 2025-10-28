import json
import hashlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature

SEED = 42
URL = "https://raw.githubusercontent.com/gchandra10/filestorage/main/FuelConsumptionCo2.csv"
EXPERIMENT_NAME = "emission_demo"
TRACKING_URI = "http://localhost:8080"

FEATURES = [
    "ENGINESIZE"
]
TARGET = "CO2EMISSIONS"

MODEL_NAME = "model"
REGISTERED_MODEL = "XGBoost_Regression_Model"

# ---------------- Data ----------------
def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    return df[[*FEATURES, TARGET]].copy()

def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=SEED)

# ---------------- Metrics ----------------
@dataclass
class Metrics:
    mae: float
    mse: float
    rmse: float
    r2: float

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
        reg_lambda=1.0,
        reg_alpha=0.0,
    )
    model.fit(X_train, y_train)
    return model

def evaluate(model: xgb.XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[np.ndarray, Metrics]:
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred)
    return y_pred, Metrics(mae, mse, rmse, r2)

# ---------------- MLflow setup & logging ----------------
def setup_mlflow(tracking_uri: str, experiment_name: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

def log_params(model: xgb.XGBRegressor) -> None:
    mlflow.log_params({
        "flavor": "xgboost",
        "features": json.dumps(FEATURES),
        "features_hash": hashlib.md5(",".join(FEATURES).encode()).hexdigest(),
        "n_features": len(FEATURES),
        "n_estimators": model.get_params()["n_estimators"],
        "max_depth": model.get_params()["max_depth"],
        "learning_rate": model.get_params()["learning_rate"],
        "subsample": model.get_params()["subsample"],
        "colsample_bytree": model.get_params()["colsample_bytree"],
        "reg_lambda": model.get_params()["reg_lambda"],
        "reg_alpha": model.get_params()["reg_alpha"],
        "split_test_size": 0.2,
        "random_state": SEED,
    })

def log_all_metrics(m: Metrics) -> None:
    mlflow.log_metrics({
        "mae": m.mae,
        "mse": m.mse,
        "rmse": m.rmse,
        "r2": m.r2,
    })

def log_pred_vs_actual_plot(y_test: pd.Series, y_pred: np.ndarray, fname: str = "pred_vs_actual.png") -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred, alpha=0.6, label="Predicted")
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=2, label="Perfect fit")
    ax.set_xlabel("Actual CO2EMISSIONS")
    ax.set_ylabel("Predicted CO2EMISSIONS")
    ax.set_title("XGBoost: Actual vs Predicted")
    ax.legend()
    mlflow.log_figure(fig, fname)
    plt.close(fig)

def log_model_and_tags(
    model: xgb.XGBRegressor,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_name: str = MODEL_NAME,
    registered_model: str | None = REGISTERED_MODEL,
):
    
    signature = infer_signature(X_train, model.predict(X_train))
    input_example = X_test.iloc[:3].copy()

    model_info = mlflow.xgboost.log_model(
        xgb_model=model,
        name=model_name,
        signature=signature,
        input_example=input_example,
        pip_requirements=[
            "mlflow>=2.15.1,<4.0",
            "xgboost>=3.0",
            "pandas>=2.2,<2.5",
            "numpy>=2.1,<2.5",
            "scikit-learn>=1.5,<1.9",
        ],
        registered_model_name=registered_model,
        model_format="json"
    )

    mlflow.set_logged_model_tags(
        model_info.model_id, {"version": "v2", "type": "xgboost", "scope": "demo"}
    )
    return model_info

def main() -> None:
    setup_mlflow(TRACKING_URI, EXPERIMENT_NAME)
    df = load_data(URL)
    X_train, X_test, y_train, y_test = split_xy(df)

    model = train_model(X_train, y_train)
    y_pred, metrics = evaluate(model, X_test, y_test)

    with mlflow.start_run() as run:
        log_all_metrics(metrics)
        log_params(model)
        # log_pred_vs_actual_plot(y_test, y_pred)
        model_info = log_model_and_tags(model, X_train, X_test)

        print("Run ID:", run.info.run_id)
        print("Logged Model URI:", model_info.model_uri)

if __name__ == "__main__":
    main()
