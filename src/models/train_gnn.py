import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from loguru import logger

from src.config import DB_PATH, NOAA_STATIONS, ACTIVE_CITY, TEST_PERIOD_START
from src.preprocessing.build_graph import build_sensor_graph
from src.models.gnn_model import STGNN

WINDOW = 7
VAL_SIZE = 60
HIDDEN = 32
HEADS = 2
N_EPOCHS = 30
LR = 1e-3
STEP_SIZE = 10
GAMMA = 0.5
BATCH_SIZE = 16
PM25_FEATURE = "pm25"
WEATHER_FEATURES = ["AWND", "TMAX", "TMIN", "PRCP"]


def load_sensor_matrix(db_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load daily PM2.5 for long-history sensors into [T, N, 1].
    Gaps are forward-filled (not zeroed) to avoid injecting false readings.

    Returns: matrix [T, N, 1], sensor_ids [N], dates [T]
    """
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

    pivot = df.pivot_table(
        index="time_stamp", columns="sensor_index",
        values=PM25_FEATURE, aggfunc="first",
    ).sort_index()

    # Forward-fill gaps, apply EMA smoothing per sensor, then zero-fill leading NaNs
    pivot = pivot.ffill()
    pivot = pivot.apply(lambda col: col.ewm(span=7, adjust=False).mean())
    pivot = pivot.fillna(0)

    sensor_ids = pivot.columns.to_numpy()
    dates = pivot.index.to_numpy()
    matrix = pivot.values[:, :, np.newaxis].astype(np.float32)  # [T, N, 1]
    return matrix, sensor_ids, dates


def load_weather_matrix(db_path: str, dates: np.ndarray) -> np.ndarray:
    """
    Return NOAA weather features aligned to `dates` as [T, F_w].
    Broadcast to all sensor nodes in train_gnn.
    """
    station = NOAA_STATIONS[ACTIVE_CITY]
    cols = ", ".join(WEATHER_FEATURES)
    with duckdb.connect(db_path, read_only=True) as con:
        df = con.execute(f"""
            SELECT date, {cols} FROM raw.weather_daily
            WHERE station = '{station}'
            ORDER BY date
        """).df()

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")[WEATHER_FEATURES]
    date_index = pd.DatetimeIndex(dates)
    df = df.reindex(date_index).ffill().fillna(0)
    return df.values.astype(np.float32)  # [T, F_w]


def make_windows(matrix: np.ndarray, window: int):
    """Sliding windows → X [samples, T, N, F], y [samples, N]."""
    X, y = [], []
    for i in range(len(matrix) - window):
        X.append(matrix[i : i + window])
        y.append(matrix[i + window, :, 0])  # pm25 at next timestep
    return np.array(X), np.array(y)


def train_gnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    # --- Load data ---
    pm25_matrix, sensor_ids, dates = load_sensor_matrix(DB_PATH)   # [T, N, 1]
    weather = load_weather_matrix(DB_PATH, dates)                    # [T, F_w]

    T, N, _ = pm25_matrix.shape
    F_w = weather.shape[1]

    # Broadcast weather to all nodes: [T, N, F_w]
    weather_broadcast = np.broadcast_to(
        weather[:, np.newaxis, :], (T, N, F_w)
    ).copy().astype(np.float32)

    # Calendar features: day_of_week [0-6] and month [1-12] — same for all nodes
    dates_dt = pd.DatetimeIndex(dates)
    calendar = np.stack([
        dates_dt.dayofweek.astype(np.float32),
        dates_dt.month.astype(np.float32),
    ], axis=1)  # [T, 2]
    calendar_broadcast = np.broadcast_to(
        calendar[:, np.newaxis, :], (T, N, 2)
    ).copy().astype(np.float32)

    # Combined matrix: [T, N, 1 + F_w + 2]
    matrix = np.concatenate([pm25_matrix, weather_broadcast, calendar_broadcast], axis=2)
    F = matrix.shape[2]

    # --- Graph ---
    with duckdb.connect(DB_PATH, read_only=True) as con:
        sensors_df = con.execute(
            "SELECT sensor_index, latitude, longitude FROM sensor_table"
        ).df()
    sensors_df = sensors_df[sensors_df["sensor_index"].isin(sensor_ids)]
    sensors_df = sensors_df.set_index("sensor_index").loc[sensor_ids].reset_index()
    edge_index, _ = build_sensor_graph(sensors_df)
    edge_index = edge_index.to(device)

    # --- Windows ---
    X, y = make_windows(matrix, WINDOW)  # [S, T, N, F], [S, N]
    split = len(X) - VAL_SIZE

    # --- Z-score normalisation using training data only ---
    train_flat = X[:split].reshape(-1, F)           # [split*T*N, F]
    feat_mean = train_flat.mean(axis=0)             # [F]
    feat_std  = train_flat.std(axis=0)
    feat_std  = np.where(feat_std > 0, feat_std, 1.0)

    X = (X - feat_mean) / feat_std
    # y is pm25 (feature index 0) at next step — normalise with same stats
    y = (y - feat_mean[0]) / feat_std[0]

    np.savez(
        "datasets/gnn_norm_stats.npz",
        feat_mean=feat_mean, feat_std=feat_std,
    )
    logger.info(f"Matrix shape: {matrix.shape}, windows: {len(X)}, split: {split}")

    # --- Tensors ---
    X_train = torch.tensor(X[:split],  dtype=torch.float32).to(device)
    y_train = torch.tensor(y[:split],  dtype=torch.float32).to(device)
    X_val   = torch.tensor(X[split:],  dtype=torch.float32).to(device)
    y_val   = torch.tensor(y[split:],  dtype=torch.float32).to(device)

    # --- Model ---
    model     = STGNN(in_features=F, hidden=HIDDEN, heads=HEADS).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # --- Training loop ---
    best_val = float("inf")
    for epoch in range(N_EPOCHS):
        model.train()
        indices    = torch.randperm(len(X_train))
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, len(X_train), BATCH_SIZE):
            batch_idx = indices[start : start + BATCH_SIZE]
            xb = X_train[batch_idx]   # [B, T, N, F]
            yb = y_train[batch_idx]   # [B, N]

            optimizer.zero_grad()
            preds = model(xb, edge_index).squeeze(-1)  # [B, N]
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val, edge_index).squeeze(-1)  # [V, N]
            val_loss = criterion(val_preds, y_val).item()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "datasets/gnn_model.pth")
            logger.info(
                f"Epoch {epoch+1}/{N_EPOCHS}  "
                f"Train Loss: {epoch_loss/n_batches:.4f}  "
                f"Val Loss: {val_loss:.4f}  ↓ best — saved"
            )
        else:
            logger.info(
                f"Epoch {epoch+1}/{N_EPOCHS}  "
                f"Train Loss: {epoch_loss/n_batches:.4f}  "
                f"Val Loss: {val_loss:.4f}"
            )
        scheduler.step()

    logger.info(f"Training complete. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    train_gnn()
