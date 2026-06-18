import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error


def build_ensemble(temporal_csv: str, spatial_csv: str) -> pd.DataFrame:
    """
    Merge temporal and spatial result CSVs, align on (Sensor Index, datetime),
    fit an OLS weighted ensemble, and return a DataFrame with predictions.

    Expected columns:
      temporal_csv: Sensor Index, datetime, y_test, y_pred
      spatial_csv:  Sensor Index, Date, Y Test, Y Pred
    """
    temporal = pd.read_csv(temporal_csv)
    spatial = pd.read_csv(spatial_csv)

    # Normalise column names so both DataFrames share the same join keys
    temporal = temporal.rename(columns={"y_test": "y_test_temporal", "y_pred": "y_pred_temporal"})
    spatial = spatial.rename(columns={
        "Date": "datetime",
        "Y Pred": "y_pred_spatial",
    })

    merged = pd.merge(
        temporal,
        spatial,
        on=["Sensor Index", "datetime"],
        suffixes=("_t", "_s"),
    )

    # Use np.isclose instead of == to handle float rounding across CSV round-trips
    merged = merged[np.isclose(merged["y_test_temporal"], merged["Y Test"], atol=0.01)]

    merged = merged.rename(columns={"y_test_temporal": "y_test"})

    X_ens = merged[["y_pred_temporal", "y_pred_spatial"]].values
    y_ens = merged["y_test"].values

    ols = LinearRegression(fit_intercept=False, positive=True).fit(X_ens, y_ens)
    merged["y_pred_ensemble"] = ols.predict(X_ens)

    return merged, ols.coef_


def ensemble_rmse(merged: pd.DataFrame) -> tuple[float, float, float]:
    """Return (rmse_temporal, rmse_spatial, rmse_ensemble)."""
    y = merged["y_test"].values
    return (
        root_mean_squared_error(y, merged["y_pred_temporal"].values),
        root_mean_squared_error(y, merged["y_pred_spatial"].values),
        root_mean_squared_error(y, merged["y_pred_ensemble"].values),
    )
