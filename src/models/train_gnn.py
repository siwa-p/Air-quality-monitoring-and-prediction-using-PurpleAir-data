import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger

from src.config import DB_PATH, NOAA_STATIONS, ACTIVE_CITY, TEST_PERIOD_START
from src.preprocessing.build_graph import build_sensor_graph
from src.models.gnn_model import STGNN

WINDOW           = 7
HORIZON          = 3
VAL_SIZE         = 60
TEST_SIZE        = 60
HIDDEN           = 32
HEADS            = 2
GRU_LAYERS       = 2
N_EPOCHS         = 100
LR               = 1e-3
LR_PATIENCE      = 5
PATIENCE         = 10
BATCH_SIZE       = 16
CLIP_GRAD        = 1.0
PM25_FEATURE     = "pm25"
WEATHER_FEATURES = ["AWND", "TMAX", "TMIN", "PRCP"]


def load_sensor_matrix(db_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load daily PM2.5 → [T, N, 1], forward-filled and EMA-smoothed per sensor."""
    with duckdb.connect(db_path, read_only=True) as con:
        long_history = con.execute(f"""
            SELECT DISTINCT sensor_index FROM raw.data_daily
            WHERE time_stamp < '{TEST_PERIOD_START}'
        """).df()["sensor_index"].tolist()

        ids_sql = ", ".join(str(s) for s in long_history)
        df = con.execute(f"""
            SELECT time_stamp, sensor_index, {PM25_FEATURE}
            FROM raw.data_daily
            WHERE sensor_index IN ({ids_sql})
            ORDER BY time_stamp, sensor_index
        """).df()

    df["time_stamp"] = df["time_stamp"].astype("datetime64[ns]")
    pivot = (
        df.pivot_table(index="time_stamp", columns="sensor_index", values=PM25_FEATURE, aggfunc="mean")
        .sort_index()
        .ffill()
        .apply(lambda col: col.ewm(span=7, adjust=False).mean())
        .fillna(0)
    )
    return (
        pivot.values[:, :, np.newaxis].astype(np.float32),
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
    pm25_matrix: np.ndarray, weather: np.ndarray, dates: np.ndarray
) -> np.ndarray:
    """Assemble [T, N, F]: [pm25, lag2, lag7, lag14, weather×4, calendar×4]."""
    T, N, _ = pm25_matrix.shape
    dates_dt = pd.DatetimeIndex(dates)

    calendar = np.stack([
        np.sin(2 * np.pi * dates_dt.dayofweek / 7),
        np.cos(2 * np.pi * dates_dt.dayofweek / 7),
        np.sin(2 * np.pi * (dates_dt.month - 1) / 12),
        np.cos(2 * np.pi * (dates_dt.month - 1) / 12),
    ], axis=1).astype(np.float32)

    lags = np.stack([_lag(pm25_matrix[:, :, 0], k) for k in (2, 7, 14)], axis=2)
    return np.concatenate([pm25_matrix, lags, _broadcast(weather, T, N), _broadcast(calendar, T, N)], axis=2)


def make_windows(matrix: np.ndarray, window: int, horizon: int):
    """Sliding windows → X [S, T, N, F], y [S, N, H]."""
    X, y = [], []
    for i in range(len(matrix) - window - horizon + 1):
        X.append(matrix[i : i + window])
        y.append(matrix[i + window : i + window + horizon, :, 0].T.copy())
    return np.array(X), np.array(y)


def train_gnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    pm25_matrix, sensor_ids, dates = load_sensor_matrix(DB_PATH)
    matrix = build_feature_matrix(pm25_matrix, load_weather_matrix(DB_PATH, dates), dates)
    _, N, F = matrix.shape

    with duckdb.connect(DB_PATH, read_only=True) as con:
        sensors_df = con.execute(
            "SELECT sensor_index, latitude, longitude FROM sensor_table"
        ).df()
    sensors_df = (
        sensors_df[sensors_df["sensor_index"].isin(sensor_ids)]
        .set_index("sensor_index").loc[sensor_ids].reset_index()
    )
    edge_index, edge_weight = build_sensor_graph(sensors_df)
    edge_weight = (edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min() + 1e-8)
    edge_index  = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    X, y = make_windows(matrix, WINDOW, HORIZON)
    n          = len(X)
    split_test = n - TEST_SIZE
    split_val  = split_test - VAL_SIZE
    assert split_val > 0, f"Not enough data: {n} windows for val={VAL_SIZE} + test={TEST_SIZE}"

    train_flat = X[:split_val].reshape(-1, F)
    feat_mean  = train_flat.mean(axis=0)
    feat_std   = np.where(train_flat.std(axis=0) > 0, train_flat.std(axis=0), 1.0)

    X = (X - feat_mean) / feat_std
    y = (y - feat_mean[0]) / feat_std[0]

    np.savez("datasets/gnn_norm_stats.npz", feat_mean=feat_mean, feat_std=feat_std)
    logger.info(f"Matrix {matrix.shape} | windows {n} | train {split_val} | val {VAL_SIZE} | test {TEST_SIZE}")

    X_train, y_train = _as_tensor(X[:split_val], device),           _as_tensor(y[:split_val], device)
    X_val,   y_val   = _as_tensor(X[split_val:split_test], device), _as_tensor(y[split_val:split_test], device)
    X_test,  y_test  = _as_tensor(X[split_test:], device),          _as_tensor(y[split_test:], device)

    persist_pred = X_test[:, -1, :, 0:1].expand_as(y_test)
    persist_rmse = ((persist_pred - y_test) ** 2).mean(dim=(0, 1)).sqrt() * feat_std[0]
    for h, rmse in enumerate(persist_rmse.tolist(), 1):
        logger.info(f"Persistence  Day+{h}: {rmse:.3f} µg/m³")

    model    = STGNN(in_features=F, hidden=HIDDEN, heads=HEADS, gru_layers=GRU_LAYERS, horizon=HORIZON).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {n_params:,} parameters")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=LR_PATIENCE, factor=0.5,
    )

    best_val         = float("inf")
    patience_counter = 0

    for epoch in range(N_EPOCHS):
        model.train()
        indices      = torch.randperm(len(X_train))
        batch_losses = []

        for start in range(0, len(X_train), BATCH_SIZE):
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
    train_gnn()
