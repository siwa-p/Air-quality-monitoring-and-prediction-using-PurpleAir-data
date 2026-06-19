import numpy as np
import pandas as pd
import torch
from sklearn.metrics import root_mean_squared_error

from src.config import DB_PATH, TEST_PERIOD_START
from src.models.train_gnn import (
    WINDOW, VAL_SIZE, HIDDEN, HEADS,
    PM25_FEATURE, WEATHER_FEATURES,
    load_sensor_matrix, load_weather_matrix, make_windows,
)
from src.models.gnn_model import STGNN
from src.preprocessing.build_graph import build_sensor_graph
import duckdb


def evaluate_gnn(
    model_path: str = "datasets/gnn_model.pth",
    norm_path: str = "datasets/gnn_norm_stats.npz",
    output_csv: str = "datasets/gnn_results.csv",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild the same matrix used during training
    pm25_matrix, sensor_ids, dates = load_sensor_matrix(DB_PATH)
    weather = load_weather_matrix(DB_PATH, dates)

    T, N, _ = pm25_matrix.shape
    F_w = weather.shape[1]

    weather_broadcast = np.broadcast_to(
        weather[:, np.newaxis, :], (T, N, F_w)
    ).copy().astype(np.float32)

    dates_dt = pd.DatetimeIndex(dates)
    calendar = np.stack([
        dates_dt.dayofweek.astype(np.float32),
        dates_dt.month.astype(np.float32),
    ], axis=1)
    calendar_broadcast = np.broadcast_to(
        calendar[:, np.newaxis, :], (T, N, 2)
    ).copy().astype(np.float32)

    matrix = np.concatenate([pm25_matrix, weather_broadcast, calendar_broadcast], axis=2)
    F = matrix.shape[2]

    # Load normalisation stats saved during training
    stats = np.load(norm_path)
    feat_mean = stats["feat_mean"]
    feat_std  = stats["feat_std"]

    X, y = make_windows(matrix, WINDOW)
    split = len(X) - VAL_SIZE

    # Apply the same z-score normalisation
    X = (X - feat_mean) / feat_std
    y = (y - feat_mean[0]) / feat_std[0]

    X_val = torch.tensor(X[split:], dtype=torch.float32).to(device)
    y_val = torch.tensor(y[split:], dtype=torch.float32).to(device)

    # Sensor graph (same k=5 as training)
    with duckdb.connect(DB_PATH, read_only=True) as con:
        sensors_df = con.execute(
            "SELECT sensor_index, latitude, longitude FROM sensor_table"
        ).df()
    sensors_df = sensors_df[sensors_df["sensor_index"].isin(sensor_ids)]
    sensors_df = sensors_df.set_index("sensor_index").loc[sensor_ids].reset_index()
    edge_index, _ = build_sensor_graph(sensors_df)
    edge_index = edge_index.to(device)

    # Load model
    model = STGNN(in_features=F, hidden=HIDDEN, heads=HEADS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        val_preds_norm = model(X_val, edge_index).squeeze(-1)  # [V, N]

    # Denormalise to µg/m³
    val_preds   = val_preds_norm.cpu().numpy() * feat_std[0] + feat_mean[0]
    val_actuals = y_val.cpu().numpy()           * feat_std[0] + feat_mean[0]

    # Dates for the validation windows
    val_dates = dates[split + WINDOW :]

    print(f"\nGNN per-sensor RMSE on validation ({VAL_SIZE} days):")
    rows = []
    for i, sensor_id in enumerate(sensor_ids):
        rmse = root_mean_squared_error(val_actuals[:, i], val_preds[:, i])
        print(f"  Sensor {sensor_id}: {rmse:.2f} µg/m³")
        for v in range(len(val_preds)):
            rows.append({
                "Date":         pd.Timestamp(val_dates[v]).date(),
                "Sensor Index": int(sensor_id),
                "Y Test":       float(val_actuals[v, i]),
                "Y Pred":       float(val_preds[v, i]),
            })

    overall_rmse = root_mean_squared_error(val_actuals.ravel(), val_preds.ravel())
    print(f"\nOverall RMSE: {overall_rmse:.2f} µg/m³")

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Saved {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    evaluate_gnn()
