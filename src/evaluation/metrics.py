import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """
    Compute a suite of regression metrics including tail-specific performance.

    Returns a dict with keys: label, n, rmse, mae, bias, mape, p95_rmse, p95_bias.

    - bias     = mean(y_pred - y_true)  — positive means systematic overestimate
    - p95_rmse = RMSE restricted to samples where y_true >= 95th percentile of y_true
    - p95_bias = bias in that same extreme subset
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) == 0:
        return {"label": label, "n": 0, "rmse": np.nan, "mae": np.nan,
                "bias": np.nan, "mape": np.nan, "p95_rmse": np.nan, "p95_bias": np.nan}

    rmse = root_mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))

    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]))) * 100 if nonzero.any() else np.nan

    p95_thresh = np.percentile(y_true, 95)
    tail = y_true >= p95_thresh
    p95_rmse = root_mean_squared_error(y_true[tail], y_pred[tail]) if tail.any() else np.nan
    p95_bias = float(np.mean(y_pred[tail] - y_true[tail])) if tail.any() else np.nan

    return {
        "label":    label,
        "n":        int(len(y_true)),
        "rmse":     round(rmse, 3),
        "mae":      round(mae, 3),
        "bias":     round(bias, 3),
        "mape":     round(mape, 2) if not np.isnan(mape) else np.nan,
        "p95_rmse": round(p95_rmse, 3) if not np.isnan(p95_rmse) else np.nan,
        "p95_bias": round(p95_bias, 3) if not np.isnan(p95_bias) else np.nan,
    }


def model_comparison_table(csv_paths: dict) -> pd.DataFrame:
    """
    Load result CSVs and compute per-sensor + overall metrics for each model.

    csv_paths: dict mapping model name → CSV path, e.g.
        {"temporal": "datasets/temporal_results.csv",
         "spatial":  "datasets/spatial_results.csv",
         "arima":    "datasets/sarimax_results.csv",
         "gnn":      "datasets/gnn_results.csv"}

    All CSVs must follow the unified schema: Date, Sensor Index, Y Test, Y Pred.

    Returns a DataFrame with columns:
        Model, Sensor Index, n, RMSE, MAE, Bias, MAPE, P95 RMSE, P95 Bias
    plus an "ALL" row per model for the overall metrics.
    """
    rows = []
    for model_name, path in csv_paths.items():
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"Skipping {model_name}: {path} not found")
            continue

        for sensor_id, grp in df.groupby("Sensor Index"):
            m = compute_metrics(grp["Y Test"].values, grp["Y Pred"].values, label=model_name)
            rows.append({"Model": model_name, "Sensor Index": sensor_id, **m})

        # Overall row across all sensors
        m_all = compute_metrics(df["Y Test"].values, df["Y Pred"].values, label=model_name)
        rows.append({"Model": model_name, "Sensor Index": "ALL", **m_all})

    result = pd.DataFrame(rows)
    result = result.rename(columns={
        "n": "N", "rmse": "RMSE", "mae": "MAE", "bias": "Bias",
        "mape": "MAPE", "p95_rmse": "P95 RMSE", "p95_bias": "P95 Bias",
    })
    result = result.drop(columns=["label"])
    return result


if __name__ == "__main__":
    import os

    paths = {
        name: f"datasets/{fname}"
        for name, fname in [
            ("temporal", "temporal_results.csv"),
            ("spatial",  "spatial_results.csv"),
            ("arima",    "sarimax_results.csv"),
            ("gnn",      "gnn_results.csv"),
        ]
        if os.path.exists(f"datasets/{fname}")
    }

    table = model_comparison_table(paths)
    all_rows = table[table["Sensor Index"] == "ALL"].drop(columns=["Sensor Index"])
    print("\n=== Overall metrics (all sensors) ===")
    print(all_rows.to_string(index=False))

    print("\n=== Per-sensor RMSE ===")
    pivot = table[table["Sensor Index"] != "ALL"].pivot_table(
        index="Sensor Index", columns="Model", values="RMSE"
    )
    print(pivot.to_string())
