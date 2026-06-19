import numpy as np
import pandas as pd
import torch
import duckdb
from loguru import logger
from sklearn.metrics import root_mean_squared_error

from src.config import DB_PATH, SPATIAL_BBOX
from src.models.train_gnn import (
    HORIZON, VAL_SIZE, TEST_SIZE,
    load_sensor_matrix, load_weather_matrix, build_feature_matrix, make_windows,
    _load_hparams,
)
from src.models.gnn_model import STGNN
from src.preprocessing.build_graph import build_sensor_graph


def evaluate_gnn(
    model_path: str = "datasets/gnn_model.pth",
    norm_path: str = "datasets/gnn_norm_stats.npz",
    output_csv: str = "datasets/gnn_results.csv",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp = _load_hparams()
    logger.info(f"Evaluating on {device} | hparams: {hp}")

    pm25_raw, pm25_smooth, sensor_ids, dates = load_sensor_matrix(DB_PATH)
    weather = load_weather_matrix(DB_PATH, dates)
    matrix = build_feature_matrix(pm25_raw, pm25_smooth, weather, dates)
    F = matrix.shape[2]

    stats = np.load(norm_path)
    feat_mean = stats["feat_mean"]
    feat_std  = stats["feat_std"]

    X, y = make_windows(matrix, hp["window"], HORIZON)

    # Read split_test saved during training to guarantee identical split
    split_test = int(stats["split_test"])

    X_norm = (X - feat_mean) / feat_std
    y_norm = (y - feat_mean[0]) / feat_std[0]

    X_test = torch.tensor(X_norm[split_test:], dtype=torch.float32).to(device)
    y_test = torch.tensor(y_norm[split_test:], dtype=torch.float32).to(device)

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

    model = STGNN(in_features=F, hidden=hp["hidden"], heads=hp["heads"],
                  gru_layers=hp["gru_layers"], horizon=HORIZON).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        preds_norm = model(X_test, edge_index, edge_weight)  # [TEST_SIZE, N, H]

    # Denormalise
    preds   = preds_norm.cpu().numpy() * feat_std[0] + feat_mean[0]   # [TEST_SIZE, N, H]
    actuals = y_test.cpu().numpy()     * feat_std[0] + feat_mean[0]   # [TEST_SIZE, N, H]

    # Report per-horizon RMSE
    for h in range(HORIZON):
        rmse_h = root_mean_squared_error(actuals[:, :, h].ravel(), preds[:, :, h].ravel())
        logger.info(f"Test  Day+{h+1} RMSE: {rmse_h:.3f} µg/m³")

    # CSV uses Day+1 (h=0) for direct comparison with other single-step models
    n_test = actuals.shape[0]
    test_dates = dates[split_test + hp["window"] : split_test + hp["window"] + n_test]

    rows = []
    per_sensor_rmse = []
    for i, sensor_id in enumerate(sensor_ids):
        rmse = root_mean_squared_error(actuals[:, i, 0], preds[:, i, 0])
        per_sensor_rmse.append(rmse)
        for t in range(len(test_dates)):
            rows.append({
                "Date":         pd.Timestamp(test_dates[t]).date(),
                "Sensor Index": int(sensor_id),
                "Y Test":       float(actuals[t, i, 0]),
                "Y Pred":       float(preds[t, i, 0]),
            })

    overall = root_mean_squared_error(actuals[:, :, 0].ravel(), preds[:, :, 0].ravel())
    logger.info(f"Overall Day+1 RMSE: {overall:.3f} µg/m³ | "
                f"median sensor: {np.median(per_sensor_rmse):.3f} µg/m³")

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    logger.info(f"Saved {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    logger.add("logs/evaluate_gnn.log", rotation="10 MB", retention=3)
    evaluate_gnn()
