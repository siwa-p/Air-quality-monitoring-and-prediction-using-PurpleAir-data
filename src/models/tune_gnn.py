import json
import duckdb
import numpy as np
import torch
import torch.nn as nn
import optuna
from loguru import logger

from src.config import DB_PATH
from src.preprocessing.build_graph import build_sensor_graph
from src.models.gnn_model import STGNN
from src.models.train_gnn import (
    load_sensor_matrix,
    load_weather_matrix,
    build_feature_matrix,
    make_windows,
    _as_tensor,
    HORIZON,
    VAL_SIZE,
    TEST_SIZE,
)

N_TRIALS   = 40
MAX_EPOCHS = 20
PATIENCE   = 5

optuna.logging.set_verbosity(optuna.logging.WARNING)


def prepare_data() -> tuple[np.ndarray, object]:
    """Load data and assemble feature matrix. Called once before tuning."""
    pm25_matrix, sensor_ids, dates = load_sensor_matrix(DB_PATH)
    matrix = build_feature_matrix(pm25_matrix, load_weather_matrix(DB_PATH, dates), dates)

    with duckdb.connect(DB_PATH, read_only=True) as con:
        sensors_df = con.execute(
            "SELECT sensor_index, latitude, longitude FROM sensor_table"
        ).df()
    sensors_df = (
        sensors_df[sensors_df["sensor_index"].isin(sensor_ids)]
        .set_index("sensor_index").loc[sensor_ids].reset_index()
    )
    return matrix, sensors_df


def objective(trial: optuna.Trial, matrix: np.ndarray, sensors_df, device: torch.device) -> float:
    lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden     = trial.suggest_categorical("hidden", [16, 32, 64])
    heads      = trial.suggest_categorical("heads", [1, 2, 4])
    dropout    = trial.suggest_float("dropout", 0.0, 0.3)
    window     = trial.suggest_categorical("window", [3, 5, 7, 14])
    k          = trial.suggest_categorical("k", [3, 5, 7, 10])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    gru_layers = trial.suggest_categorical("gru_layers", [1, 2])

    edge_index, edge_weight = build_sensor_graph(sensors_df, k=k)
    edge_weight = (edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min() + 1e-8)
    edge_index  = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    X, y = make_windows(matrix, window, HORIZON)
    n          = len(X)
    split_test = n - TEST_SIZE
    split_val  = split_test - VAL_SIZE

    if split_val <= 0:
        raise optuna.TrialPruned()

    F = matrix.shape[2]
    train_flat = X[:split_val].reshape(-1, F)
    feat_mean  = train_flat.mean(axis=0)
    feat_std   = np.where(train_flat.std(axis=0) > 0, train_flat.std(axis=0), 1.0)

    X = (X - feat_mean) / feat_std
    y = (y - feat_mean[0]) / feat_std[0]

    X_train, y_train = _as_tensor(X[:split_val], device),           _as_tensor(y[:split_val], device)
    X_val,   y_val   = _as_tensor(X[split_val:split_test], device), _as_tensor(y[split_val:split_test], device)

    model     = STGNN(in_features=F, hidden=hidden, heads=heads, dropout=dropout,
                      gru_layers=gru_layers, horizon=HORIZON).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val   = float("inf")
    no_improve = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        indices = torch.randperm(len(X_train))
        for start in range(0, len(X_train), batch_size):
            idx = indices[start : start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_train[idx], edge_index, edge_weight), y_train[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val, edge_index, edge_weight), y_val).item()

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    return best_val


def tune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Tuning on {device} | {N_TRIALS} trials | {MAX_EPOCHS} epochs max per trial")

    matrix, sensors_df = prepare_data()

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )
    study.optimize(
        lambda trial: objective(trial, matrix, sensors_df, device),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best = study.best_params
    logger.info(f"Best val loss: {study.best_value:.4f}")
    logger.info(f"Best params: {best}")

    with open("datasets/gnn_best_params.json", "w") as f:
        json.dump(best, f, indent=2)
    logger.info("Saved → datasets/gnn_best_params.json")

    logger.info("Update train_gnn.py with:")
    for key, val in best.items():
        logger.info(f"  {key.upper():<12} = {val!r}")


if __name__ == "__main__":
    tune()
