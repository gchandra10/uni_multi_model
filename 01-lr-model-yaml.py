import json
import hashlib
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


# ---------------- Config ----------------
@dataclass
class Config:
    data: Dict[str, Any]
    mlflow: Dict[str, Any]
    model: Dict[str, Any]
    requirements: list
    plot: Dict[str, Any]
    preprocess: Dict[str, Any]


def load_config(path: str = "config.yaml") -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(
        data=raw["data"],
        mlflow=raw["mlflow"],
        model=raw["model"],
        requirements=raw["requirements"],
        plot=raw["plot"],
        preprocess=raw["preprocess"],
    )


# ---------------- Data ----------------
def load_data(url: str, features: list, target: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    return df[[*features, target]].copy()


def split_xy(
    df: pd.DataFrame, features: list, target: str, test_size: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


# ---------------- Metrics ----------------
@dataclass
class Metrics:
    mae: float
    mse: float
    rmse: float
    r2: float


def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_cfg: Dict[str, Any]) -> LinearRegression:
    if model_cfg["type"] != "LinearRegression":
        raise ValueError("Only LinearRegression is supported in this example. Update train_model to add more.")
    params = model_cfg.get("params", {})
    model = LinearRegression(**params)
    model.fit(X_train, y_train)
    return model


def evaluate(model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[np.ndarray, Metrics]:
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred)
    return y_pred, Metrics(mae, mse, rmse, r2)


def setup_mlflow(tracking_uri: str, experiment_name: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_params(cfg: Config, features: list) -> None:
    mlflow.log_params(
        {
            "model": cfg.model["type"],
            "features": json.dumps(features),
            "features_hash": hashlib.md5(",".join(features).encode()).hexdigest(),
            "n_features": len(features),
            "fit_intercept": cfg.model["params"].get("fit_intercept", True),
            "positive": cfg.model["params"].get("positive", False),
            "split_test_size": cfg.data["test_size"],
            "random_state": cfg.data["random_state"],
            "imputer_num": cfg.preprocess.get("imputer_num", "n.a."),
            "scaler_num": cfg.preprocess.get("scaler_num", "n.a."),
        }
    )


def log_all_metrics(m: Metrics) -> None:
    mlflow.log_metrics({"mae": m.mae, "mse": m.mse, "rmse": m.rmse, "r2": m.r2})


def log_pred_vs_actual_plot(
    y_test: pd.Series, y_pred: np.ndarray, plot_cfg: Dict[str, Any]
) -> None:
    fname = plot_cfg.get("filename", "pred_vs_actual.png")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred, alpha=0.6, label="Predicted")
    if plot_cfg.get("diagonal", True):
        lo = min(float(y_test.min()), float(y_pred.min()))
        hi = max(float(y_test.max()), float(y_pred.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=2, label="Perfect fit")
    ax.set_xlabel(plot_cfg.get("xlabel", "Actual"))
    ax.set_ylabel(plot_cfg.get("ylabel", "Predicted"))
    ax.set_title(plot_cfg.get("title", "Predicted vs Actual"))
    ax.legend()
    mlflow.log_figure(fig, fname)
    plt.close(fig)


def log_model_and_tags(
    cfg: Config,
    model: LinearRegression,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> mlflow.models.model.ModelInfo:
    signature = infer_signature(X_train, model.predict(X_train))
    input_example = X_test.iloc[:3].copy()
    model_name = cfg.mlflow["model_name"]
    registered_model: Optional[str] = cfg.mlflow.get("registered_model")

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name=model_name,
        signature=signature,
        input_example=input_example,
        pip_requirements=cfg.requirements,
        registered_model_name=registered_model if registered_model else None,
    )

    tags = cfg.mlflow.get("tags", {})
    if tags:
        mlflow.set_logged_model_tags(model_info.model_id, {str(k): str(v) for k, v in tags.items()})

    return model_info


def main() -> None:
    cfg = load_config("config.yaml")

    setup_mlflow(cfg.mlflow["tracking_uri"], cfg.mlflow["experiment_name"])

    df = load_data(cfg.data["url"], cfg.data["features"], cfg.data["target"])
    X_train, X_test, y_train, y_test = split_xy(
        df,
        cfg.data["features"],
        cfg.data["target"],
        cfg.data["test_size"],
        cfg.data["random_state"],
    )

    model = train_model(X_train, y_train, cfg.model)
    y_pred, metrics = evaluate(model, X_test, y_test)

    with mlflow.start_run():
        log_all_metrics(metrics)
        log_params(cfg, cfg.data["features"])
        log_pred_vs_actual_plot(y_test, y_pred, cfg.plot)
        model_info = log_model_and_tags(cfg, model, X_train, X_test)

        run = mlflow.active_run()
        print("Run ID:", run.info.run_id if run else "n.a.")
        print("Logged Model URI:", model_info.model_uri)


if __name__ == "__main__":
    main()
