import json
import os
import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger

from src.config import DB_PATH, NOAA_STATIONS, ACTIVE_CITY, TEST_PERIOD_START, SPATIAL_BBOX, PM25_OUTLIER_THRESHOLD
from src.preprocessing.build_graph import build_sensor_graph
from src.preprocessing.sensor_filter import apply_hampel_filter
from src.models.gnn_model import STGNN

# --- Fixed architecture / training constants (not tuned by Optuna) ---
HORIZON          = 3
VAL_SIZE         = 60
TEST_SIZE        = 60
N_EPOCHS         = 100
LR_PATIENCE      = 5
PATIENCE         = 10
CLIP_GRAD        = 1.0
PM25_FEATURE     = "pm25"
WEATHER_FEATURES = ["AWND", "TMAX", "TMIN", "PRCP"]

# --- Defaults for Optuna-tunable hyperparams (overridden by gnn_best_params.json) ---
WINDOW      = 14
HIDDEN      = 16
HEADS       = 4
DROPOUT     = 0.064
GRU_LAYERS  = 2
LR          = 6.963e-3
BATCH_SIZE  = 16
K_NEIGHBORS = 3

BEST_PARAMS_PATH = "datasets/gnn_best_params.json"


def _load_hparams() -> dict:
    """Return hyperparams from gnn_best_params.json, falling back to module defaults."""
    defaults = dict(window=WINDOW, hidden=HIDDEN, heads=HEADS, dropout=DROPOUT,
                    gru_layers=GRU_LAYERS, lr=LR, batch_size=BATCH_SIZE, k=K_NEIGHBORS)
    if os.path.exists(BEST_PARAMS_PATH):
        with open(BEST_PARAMS_PATH) as f:
            saved = json.load(f)
        defaults.update({k: saved[k] for k in defaults if k in saved})
        logger.info(f"Loaded hyperparams from {BEST_PARAMS_PATH}")
    else:
        logger.info("No gnn_best_params.json found — using module defaults")
    return defaults


def load_sensor_matrix(db_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load daily PM2.5 → raw [T,N,1] and smooth [T,N,1] arrays.

    Returns (pm25_raw, pm25_smooth, sensor_ids, dates).
    - pm25_raw:    Hampel-filtered (spike-cleaned) — used as feature 0 and prediction target.
    - pm25_smooth: EMA span=3 of pm25_raw — used for lag features only (indices 1-3).
    """
    bb = SPATIAL_BBOX
    with duckdb.connect(db_path, read_only=True) as con:
        long_history = con.execute(f"""
            SELECT DISTINCT d.sensor_index FROM raw.data_daily d
            JOIN sensor_table s ON s.sensor_index = d.sensor_index
            WHERE d.time_stamp < '{TEST_PERIOD_START}'
              AND s.longitude BETWEEN {bb['lon_min']} AND {bb['lon_max']}
              AND s.latitude  BETWEEN {bb['lat_min']} AND {bb['lat_max']}
        """).df()["sensor_index"].tolist()

        ids_sql = ", ".join(str(s) for s in long_history)
        df = con.execute(f"""
            SELECT time_stamp, sensor_index, {PM25_FEATURE}
            FROM raw.data_daily
            WHERE sensor_index IN ({ids_sql})
              AND {PM25_FEATURE} < {PM25_OUTLIER_THRESHOLD}
            ORDER BY time_stamp, sensor_index
        """).df()

    df["time_stamp"] = df["time_stamp"].astype("datetime64[ns]")
    pivot = (
        df.pivot_table(index="time_stamp", columns="sensor_index", values=PM25_FEATURE, aggfunc="mean")
        .sort_index()
        .ffill()
        .fillna(0)
    )
    # Hampel filter per sensor column: removes spikes, preserves real events
    pivot_raw = pivot.apply(
        lambda col: apply_hampel_filter(col.reset_index(drop=True), window=3, k=3.0).values
    )
    # EMA span=3 per sensor: smooth version used only for lag features
    pivot_smooth = pivot_raw.apply(lambda col: col.ewm(span=3, adjust=False).mean())

    return (
        pivot_raw.values[:, :, np.newaxis].astype(np.float32),
        pivot_smooth.values[:, :, np.newaxis].astype(np.float32),
        pivot.columns.to_numpy(),
        pivot.index.to_numpy(),
    )


def load_weather_matrix(db_path: str, dates: np.ndarray) -> np.ndarray:
    """Return NOAA weather features aligned to dates: [T, F_w]."""
    station = NOAA_STATIONS[ACTIVE_CITY]
    cols = ", ".join(WEATHER_FEATURES)
    with duckdb.connect(db_path, read_only=True) as con:
        df = con.execute(f"""
            SELECT date, {cols} FROM raw.weather_daily
            WHERE station = '{station}'
            ORDER BY date
        """).df()
    df["date"] = pd.to_datetime(df["date"])
    return (
        df.set_index("date")[WEATHER_FEATURES]
        .reindex(pd.DatetimeIndex(dates))
        .ffill()
        .fillna(0)
        .values.astype(np.float32)
    )


def _lag(arr: np.ndarray, k: int) -> np.ndarray:
    if k == 0:
        return arr.copy()
    out = np.zeros_like(arr)
    out[k:] = arr[:-k]
    return out


def _as_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32).to(device)


def _broadcast(arr: np.ndarray, T: int, N: int) -> np.ndarray:
    # .copy() required: broadcast_to returns a read-only view; normalization writes to it later
    return np.broadcast_to(arr[:, np.newaxis, :], (T, N, arr.shape[1])).copy().astype(np.float32)


def build_feature_matrix(
    pm25_raw: np.ndarray, pm25_smooth: np.ndarray, weather: np.ndarray, dates: np.ndarray
) -> np.ndarray:
    """Assemble [T, N, F=12]: [pm25_raw, lag2_smooth, lag7_smooth, lag14_smooth, weather×4, calendar×4].

    pm25_raw   → feature 0 and prediction target (Hampel-cleaned, no EMA).
    pm25_smooth → lag features 1-3 (EMA span=3 of pm25_raw, gives smoother lag signal).
    """
    T, N, _ = pm25_raw.shape
    dates_dt = pd.DatetimeIndex(dates)

    calendar = np.stack([
        np.sin(2 * np.pi * dates_dt.dayofweek / 7),
        np.cos(2 * np.pi * dates_dt.dayofweek / 7),
        np.sin(2 * np.pi * (dates_dt.month - 1) / 12),
        np.cos(2 * np.pi * (dates_dt.month - 1) / 12),
    ], axis=1).astype(np.float32)

    lags = np.stack([_lag(pm25_smooth[:, :, 0], k) for k in (2, 7, 14)], axis=2)
    return np.concatenate([pm25_raw, lags, _broadcast(weather, T, N), _broadcast(calendar, T, N)], axis=2)


def make_windows(matrix: np.ndarray, window: int, horizon: int):
    """Sliding windows → X [S, T, N, F], y [S, N, H]."""
    X, y = [], []
    for i in range(len(matrix) - window - horizon + 1):
        X.append(matrix[i : i + window])
        y.append(matrix[i + window : i + window + horizon, :, 0].T.copy())
    return np.array(X), np.array(y)


def train_gnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp = _load_hparams()
    logger.info(f"Training on {device} | hparams: {hp}")

    pm25_raw, pm25_smooth, sensor_ids, dates = load_sensor_matrix(DB_PATH)
    matrix = build_feature_matrix(pm25_raw, pm25_smooth, load_weather_matrix(DB_PATH, dates), dates)
    _, N, F = matrix.shape

    with duckdb.connect(DB_PATH, read_only=True) as con:
        sensors_df = con.execute(
            "SELECT sensor_index, latitude, longitude FROM sensor_table"
        ).df()
    sensors_df = (
        sensors_df[sensors_df["sensor_index"].isin(sensor_ids)]
        .set_index("sensor_index").loc[sensor_ids].reset_index()
    )
    edge_index, edge_weight = build_sensor_graph(sensors_df, k=hp["k"])
    edge_weight = (edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min() + 1e-8)
    edge_index  = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    X, y = make_windows(matrix, hp["window"], HORIZON)
    n = len(X)

    # Date-based split: align with TEST_PERIOD_START so GNN test period matches
    # temporal/spatial. Window i predicts dates[i+window] onward (Day+1).
    dates_dt   = pd.DatetimeIndex(dates)
    pred_dates = dates_dt[hp["window"] : hp["window"] + n]
    test_start = pd.Timestamp(TEST_PERIOD_START)
    split_test = int(np.searchsorted(pred_dates, test_start))
    split_val  = split_test - VAL_SIZE
    assert split_val > 0, (
        f"Not enough pre-test data: split_test={split_test}, val={VAL_SIZE}"
    )
    logger.info(
        f"Split: train={split_val} | val={VAL_SIZE} | test={n - split_test} "
        f"(test from {pred_dates[split_test].date()})"
    )

    train_flat = X[:split_val].reshape(-1, F)
    feat_mean  = train_flat.mean(axis=0)
    feat_std   = np.where(train_flat.std(axis=0) > 0, train_flat.std(axis=0), 1.0)

    X = (X - feat_mean) / feat_std
    y = (y - feat_mean[0]) / feat_std[0]

    np.savez("datasets/gnn_norm_stats.npz", feat_mean=feat_mean, feat_std=feat_std,
             split_test=split_test, window=hp["window"])
    logger.info(f"Matrix {matrix.shape} | windows {n} | feat_std[pm25]={feat_std[0]:.3f}")

    X_train, y_train = _as_tensor(X[:split_val], device),           _as_tensor(y[:split_val], device)
    X_val,   y_val   = _as_tensor(X[split_val:split_test], device), _as_tensor(y[split_val:split_test], device)
    X_test,  y_test  = _as_tensor(X[split_test:], device),          _as_tensor(y[split_test:], device)

    persist_pred = X_test[:, -1, :, 0:1].expand_as(y_test)
    persist_rmse = ((persist_pred - y_test) ** 2).mean(dim=(0, 1)).sqrt() * feat_std[0]
    for h, rmse in enumerate(persist_rmse.tolist(), 1):
        logger.info(f"Persistence  Day+{h}: {rmse:.3f} µg/m³")

    model    = STGNN(in_features=F, hidden=hp["hidden"], heads=hp["heads"],
                     dropout=hp["dropout"], gru_layers=hp["gru_layers"], horizon=HORIZON).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {n_params:,} parameters")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=LR_PATIENCE, factor=0.5,
    )

    best_val         = float("inf")
    patience_counter = 0

    for epoch in range(N_EPOCHS):
        model.train()
        indices      = torch.randperm(len(X_train))
        batch_losses = []

        for start in range(0, len(X_train), hp["batch_size"]):
            idx = indices[start : start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(X_train[idx], edge_index, edge_weight), y_train[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()
            batch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val, edge_index, edge_weight), y_val).item()

        tag = ""
        if val_loss < best_val:
            best_val         = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "datasets/gnn_model.pth")
            tag = "  ↓ saved"
        else:
            patience_counter += 1
            tag = f"  [{patience_counter}/{PATIENCE}]"

        logger.info(f"Epoch {epoch+1:02d}/{N_EPOCHS}  train {np.mean(batch_losses):.4f}  val {val_loss:.4f}{tag}")
        scheduler.step(val_loss)

        if patience_counter >= PATIENCE:
            logger.info("Early stopping")
            break

    model.load_state_dict(torch.load("datasets/gnn_model.pth", map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test, edge_index, edge_weight)

    per_h_rmse = ((test_preds - y_test) ** 2).mean(dim=(0, 1)).sqrt() * feat_std[0]
    for h, rmse in enumerate(per_h_rmse.tolist(), 1):
        logger.info(f"GNN  Day+{h} RMSE: {rmse:.3f} µg/m³")


if __name__ == "__main__":
    logger.add("logs/gnn_train.log", rotation="10 MB", retention=3)
    train_gnn()
