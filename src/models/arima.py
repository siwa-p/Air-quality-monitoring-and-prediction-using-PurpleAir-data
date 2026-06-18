import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.config import TEST_PERIOD_START

DROP_COLS = ["name", "latitude", "longitude", "STATION", "LATITUDE", "LONGITUDE", "ELEVATION"]


def run_arima(merged_csv: str = "datasets/merged_data.csv",
              output_csv: str = "datasets/sarimax_results.csv") -> pd.DataFrame:
    merged_data = pd.read_csv(merged_csv, index_col=False)
    merged_data["time_stamp"] = pd.to_datetime(merged_data["time_stamp"])
    merged_data.set_index("time_stamp", inplace=True)
    merged_data.sort_index(inplace=True)
    merged_data.index = merged_data.index.to_period("D")

    sensor_results = []
    for sensor_index in merged_data["sensor_index"].unique():
        sensor_data = merged_data[merged_data["sensor_index"] == sensor_index]
        if sensor_data.empty:
            continue

        latitude = sensor_data["latitude"].iloc[0]
        longitude = sensor_data["longitude"].iloc[0]
        train_data = sensor_data[:TEST_PERIOD_START]
        test_data = sensor_data[TEST_PERIOD_START:]

        if train_data.empty or test_data.empty:
            print(f"Insufficient data for sensor {sensor_index}")
            continue

        drop_existing = [c for c in DROP_COLS + ["sensor_index"] if c in sensor_data.columns]
        y = train_data["pm25"]
        X = train_data.drop(columns=["pm25"] + drop_existing)
        X_test = test_data.drop(columns=["pm25"] + drop_existing)

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
        X = X.fillna(X.mean())
        X_test = X_test.fillna(X_test.mean())

        if X.isnull().values.any() or X_test.isnull().values.any():
            print(f"NaNs in exogenous data for sensor {sensor_index}, skipping.")
            continue

        try:
            model_fit = SARIMAX(y, exog=X, order=(1, 0, 1)).fit(
                maxiter=5000, method="bfgs", disp=False
            )
        except Exception as e:
            print(f"Fitting failed for sensor {sensor_index}: {e}")
            continue

        try:
            predictions = model_fit.predict(
                start=len(train_data),
                end=len(train_data) + len(test_data) - 1,
                exog=X_test,
            )
            if predictions.isna().any():
                predictions = np.nan

            sensor_results.append({
                "Sensor Index": sensor_index,
                "Latitude": latitude,
                "Longitude": longitude,
                "y_test": test_data["pm25"].tolist(),
                "y_pred": predictions.tolist() if not isinstance(predictions, float) else np.nan,
            })
        except Exception as e:
            print(f"Prediction failed for sensor {sensor_index}: {e}")
            continue

    results_df = pd.DataFrame(sensor_results)
    results_df.to_csv(output_csv, index=False)
    print(f"Saved {len(results_df)} sensor results to {output_csv}")
    return results_df


if __name__ == "__main__":
    run_arima()
