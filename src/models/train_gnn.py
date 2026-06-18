import duckdb
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data

from src.config import DB_PATH, ACTIVE_CITY, TEST_PERIOD_START
from src.preprocessing.build_graph import build_sensor_graph
from src.models.gnn_model import STGNN

WINDOW = 14        # timesteps per sample
VAL_SIZE = 60      # hold-out last N time points for validation
HIDDEN = 64
HEADS = 4
N_EPOCHS = 50
LR = 1e-3
STEP_SIZE = 20
GAMMA = 0.5
BATCH_SIZE = 32
FEATURES = ["pm25"]


def load_sensor_matrix(db_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load daily sensor readings from DuckDB and pivot into [T, N, F] array.

    Returns:
        matrix: np.ndarray [T, N, F]
        sensor_ids: np.ndarray [N]
    """
    with duckdb.connect(db_path, read_only=True) as con:
        df = con.execute(f"""
            SELECT d.time_stamp, d.sensor_index, {", ".join(f"d.{c}" for c in FEATURES)}
            FROM raw.data_daily AS d
            ORDER BY d.time_stamp, d.sensor_index
        """).df()

    df["time_stamp"] = df["time_stamp"].astype("datetime64[ns]")
    df = df.sort_values(["time_stamp", "sensor_index"])

    sensor_ids = df["sensor_index"].unique()
    dates = df["time_stamp"].unique()

    N = len(sensor_ids)
    T = len(dates)
    F = len(FEATURES)

    sid_idx = {s: i for i, s in enumerate(sensor_ids)}
    date_idx = {d: i for i, d in enumerate(sorted(dates))}

    matrix = np.zeros((T, N, F), dtype=np.float32)
    for _, row in df.iterrows():
        t = date_idx[row["time_stamp"]]
        n = sid_idx[row["sensor_index"]]
        matrix[t, n] = [row[c] for c in FEATURES]

    return matrix, sensor_ids


def make_windows(matrix: np.ndarray, window: int):
    """Sliding windows → X [samples, T, N, F], y [samples, N]."""
    X, y = [], []
    for i in range(len(matrix) - window):
        X.append(matrix[i:i + window])
        y.append(matrix[i + window, :, 0])   # pm2_5 (index 0) at next step
    return np.array(X), np.array(y)


def train_gnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    matrix, sensor_ids = load_sensor_matrix(DB_PATH)

    with duckdb.connect(DB_PATH, read_only=True) as con:
        sensors_df = con.execute(
            "SELECT sensor_index, latitude, longitude FROM sensor_table"
        ).df()
    sensors_df = sensors_df[sensors_df["sensor_index"].isin(sensor_ids)]
    sensors_df = sensors_df.set_index("sensor_index").loc[sensor_ids].reset_index()

    edge_index, edge_weight = build_sensor_graph(sensors_df)
    edge_index = edge_index.to(device)

    X, y = make_windows(matrix, WINDOW)

    split = len(X) - VAL_SIZE
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

    N, F = matrix.shape[1], matrix.shape[2]
    model = STGNN(in_features=F, hidden=HIDDEN, heads=HEADS).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    for epoch in range(N_EPOCHS):
        model.train()
        indices = torch.randperm(len(X_train))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(X_train), BATCH_SIZE):
            batch_idx = indices[start:start + BATCH_SIZE]
            xb = X_train[batch_idx]   # [B, T, N, F]
            yb = y_train[batch_idx]   # [B, N]

            optimizer.zero_grad()
            batch_preds = []
            for i in range(len(xb)):
                out = model(xb[i], edge_index)   # [N, 1]
                batch_preds.append(out.squeeze(-1))
            preds = torch.stack(batch_preds)      # [B, N]
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_preds = torch.stack([
                model(X_val[i], edge_index).squeeze(-1) for i in range(len(X_val))
            ])
            val_loss = criterion(val_preds, y_val).item()

        print(
            f"Epoch {epoch+1}/{N_EPOCHS}  "
            f"Train Loss: {epoch_loss/n_batches:.4f}  "
            f"Val Loss: {val_loss:.4f}"
        )
        scheduler.step()

    torch.save(model.state_dict(), "datasets/gnn_model.pth")
    print("Saved datasets/gnn_model.pth")


if __name__ == "__main__":
    train_gnn()
