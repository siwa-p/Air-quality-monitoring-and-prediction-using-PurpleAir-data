import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error


def build_ensemble(temporal_csv: str, spatial_csv: str, gnn_csv: str = None) -> tuple:
    """
    Merge result CSVs on (Sensor Index, Date), fit a non-negative OLS ensemble.

    All CSVs must use the unified schema: Date, Sensor Index, Y Test, Y Pred.
    GNN CSV is optional; when provided a 3-model ensemble is fitted.

    Returns: (merged DataFrame, OLS weights array)
    """
    temporal = pd.read_csv(temporal_csv)
    spatial  = pd.read_csv(spatial_csv)

    temporal = temporal.rename(columns={"Y Test": "y_test_t", "Y Pred": "y_pred_temporal"})
    spatial  = spatial.rename( columns={"Y Test": "y_test_s", "Y Pred": "y_pred_spatial"})

    merged = pd.merge(temporal, spatial, on=["Sensor Index", "Date"])

    # Confirm actuals agree across CSV round-trips
    merged = merged[np.isclose(merged["y_test_t"], merged["y_test_s"], atol=0.01)]
    merged = merged.rename(columns={"y_test_t": "y_test"}).drop(columns=["y_test_s"])

    pred_cols = ["y_pred_temporal", "y_pred_spatial"]

    if gnn_csv is not None:
        gnn = pd.read_csv(gnn_csv)
        gnn = gnn.rename(columns={"Y Test": "y_test_g", "Y Pred": "y_pred_gnn"})
        merged = pd.merge(merged, gnn[["Sensor Index", "Date", "y_pred_gnn"]], on=["Sensor Index", "Date"], how="inner")
        pred_cols.append("y_pred_gnn")

    X_ens = merged[pred_cols].values
    y_ens = merged["y_test"].values

    ols = LinearRegression(fit_intercept=False, positive=True).fit(X_ens, y_ens)
    merged["y_pred_ensemble"] = ols.predict(X_ens)

    return merged, ols.coef_


def ensemble_rmse(merged: pd.DataFrame) -> dict:
    """Return per-model RMSE dict. Keys depend on which models are in `merged`."""
    y = merged["y_test"].values
    results = {}
    for col in ["y_pred_temporal", "y_pred_spatial", "y_pred_gnn", "y_pred_ensemble"]:
        if col in merged.columns:
            results[col.replace("y_pred_", "")] = root_mean_squared_error(y, merged[col].values)
    return results
