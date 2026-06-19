import pandas as pd
import numpy as np
from loguru import logger
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.config import TEST_PERIOD_START, PM25_OUTLIER_THRESHOLD, SPATIAL_BBOX
from src.preprocessing.sensor_filter import apply_hampel_filter

DROP_COLS = ["name", "latitude", "longitude", "STATION", "LATITUDE", "LONGITUDE", "ELEVATION"]


def run_arima(merged_csv: str = "datasets/merged_data.csv",
              output_csv: str = "datasets/sarimax_results.csv") -> pd.DataFrame:
    merged_data = pd.read_csv(merged_csv, index_col=False)
    bb = SPATIAL_BBOX
    merged_data = merged_data[
        merged_data["longitude"].between(bb["lon_min"], bb["lon_max"]) &
        merged_data["latitude"].between(bb["lat_min"], bb["lat_max"])
    ]
    merged_data = merged_data[merged_data["pm25"] < PM25_OUTLIER_THRESHOLD]
    merged_data["time_stamp"] = pd.to_datetime(merged_data["time_stamp"])
    merged_data.set_index("time_stamp", inplace=True)
    merged_data.sort_index(inplace=True)
    merged_data.index = merged_data.index.to_period("D")

    sensor_results = []
    for sensor_index in merged_data["sensor_index"].unique():
        sensor_data = merged_data[merged_data["sensor_index"] == sensor_index].copy()
        if sensor_data.empty:
            continue
        # Hampel filter: clean sensor spikes before model fitting
        sensor_data["pm25"] = apply_hampel_filter(
            sensor_data["pm25"].reset_index(drop=True), window=3, k=3.0
        ).values

        latitude = sensor_data["latitude"].iloc[0]
        longitude = sensor_data["longitude"].iloc[0]
        train_data = sensor_data[:TEST_PERIOD_START]
        test_data = sensor_data[TEST_PERIOD_START:]

        if train_data.empty or test_data.empty:
            logger.warning(f"Insufficient data for sensor {sensor_index}")
            continue

        drop_existing = [c for c in DROP_COLS + ["sensor_index"] if c in sensor_data.columns]
        y = train_data["pm25"]
        X = train_data.drop(columns=["pm25"] + drop_existing)
        X_test = test_data.drop(columns=["pm25"] + drop_existing)

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Drop columns that are entirely NaN in training (mean can't fill them)
        all_nan_cols = X.columns[X.isna().all()].tolist()
        X = X.drop(columns=all_nan_cols)
        X_test = X_test.drop(columns=all_nan_cols)
        X = X.fillna(X.mean())
        X_test = X_test.fillna(X_test.mean())

        if X.isnull().values.any() or X_test.isnull().values.any():
            logger.warning(f"NaNs in exogenous data for sensor {sensor_index}, skipping.")
            continue

        try:
            model_fit = SARIMAX(y, exog=X, order=(2, 0, 1)).fit(
                maxiter=5000, method="bfgs", disp=False
            )
        except Exception as e:
            logger.warning(f"Fitting failed for sensor {sensor_index}: {e}")
            continue

        try:
            predictions = model_fit.predict(
                start=len(train_data),
                end=len(train_data) + len(test_data) - 1,
                exog=X_test,
            )
            if predictions.isna().any():
                logger.warning(f"Predictions contain NaN for sensor {sensor_index}, skipping.")
                continue
        except Exception as e:
            logger.warning(f"Prediction failed for sensor {sensor_index}: {e}")
            continue

        for date, actual, pred in zip(
            test_data.index.to_timestamp(),
            test_data["pm25"].values,
            predictions.values,
        ):
            sensor_results.append({
                "Date":         date.date(),
                "Sensor Index": sensor_index,
                "Latitude":     latitude,
                "Longitude":    longitude,
                "Y Test":       actual,
                "Y Pred":       float(pred),
            })

    results_df = pd.DataFrame(sensor_results)
    results_df.to_csv(output_csv, index=False)
    logger.info(f"Saved {len(results_df)} sensor results to {output_csv}")
    return results_df


if __name__ == "__main__":
    logger.add("logs/arima_run.log", rotation="10 MB", retention=3)
    run_arima()
