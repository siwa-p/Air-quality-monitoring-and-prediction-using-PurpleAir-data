import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

from src.config import DB_PATH, NOAA_STATIONS, ACTIVE_CITY, TEST_PERIOD_START, PM25_OUTLIER_THRESHOLD
from src.preprocessing.merge_data import load_weather_df
from src.preprocessing.sensor_filter import apply_ema_filter

WEATHER_FEATURES = ["AWND", "PRCP", "SNOW", "TMAX", "TMIN", "WSF2", "WDF2"]
SPATIAL_FEATURES = ["spatial_lag_pm25"] + WEATHER_FEATURES


def calculate_spatial_weights(sensors: pd.DataFrame) -> pd.DataFrame:
    coordinates = np.column_stack((sensors["latitude"], sensors["longitude"]))
    nn = NearestNeighbors(n_neighbors=6, algorithm="kd_tree")
    nn.fit(coordinates)
    distances, neighbors = nn.kneighbors(coordinates)
    neighbors = neighbors[:, 1:]
    distances = distances[:, 1:]

    sensors_idx = sensors.set_index("sensor_index")
    weights = 1 / distances
    weights_dict = {}
    for i, sid in enumerate(sensors_idx.index):
        neighbor_ids = sensors_idx.index[neighbors[i]]
        weights_dict[sid] = pd.Series(weights[i], index=neighbor_ids)

    spatial_weights = pd.DataFrame(weights_dict).fillna(0).T
    spatial_weights = spatial_weights.reindex(
        index=sensors_idx.index, columns=sensors_idx.index, fill_value=0
    )
    return spatial_weights


def get_train_test_data_for_sensor(
    data: pd.DataFrame, sensor_index: int, spatial_weights: pd.DataFrame
):
    data = data.copy()
    data["pm25"] = data["pm25"].fillna(0)
    data["spatial_lag_pm25"] = spatial_weights.values @ data["pm25"].values

    train_data = data[data.index != sensor_index]
    test_data  = data[data.index == sensor_index]

    feature_cols = [c for c in SPATIAL_FEATURES if c in data.columns and data[c].notna().any()]
    X_train = train_data[feature_cols]
    y_train = train_data["pm25"]
    X_test  = test_data[feature_cols]
    y_test  = test_data["pm25"]

    return X_train, X_test, y_train, y_test


def get_data_all(start_date: str, end_date: str) -> pd.DataFrame:
    query = f"""
    SELECT
        s.sensor_index, s.name, s.latitude, s.longitude,
        d.time_stamp, d.pm25
    FROM sensor_table AS s
    JOIN raw.data_daily AS d ON s.sensor_index = d.sensor_index
    WHERE d.time_stamp BETWEEN '{start_date}T00:00:00Z' AND '{end_date}T23:59:59Z'
    """
    with duckdb.connect(DB_PATH, read_only=True) as con:
        data = con.execute(query).df()

    data = data[data["pm25"] < PM25_OUTLIER_THRESHOLD]
    data["time_stamp"] = pd.to_datetime(data["time_stamp"])
    data.set_index("time_stamp", inplace=True)
    data.index = data.index.date
    data = data.sort_index(ascending=False)
    return data


def merge_weather(data: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join NOAA weather columns onto a date-indexed sensor DataFrame."""
    weather_df = weather_df[[c for c in WEATHER_FEATURES if c in weather_df.columns]].copy()
    weather_df.index = pd.to_datetime(weather_df.index).date
    return data.join(weather_df, how="left")


def train_evaluate_spatial(X_train, X_test, y_train, y_test) -> dict:
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("xgbreg", XGBRegressor(n_estimators=100, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return {
        "y_test": y_test,
        "y_pred": y_pred,
        "mae":  mean_absolute_error(y_test, y_pred),
        "rmse": root_mean_squared_error(y_test, y_pred),
    }


if __name__ == "__main__":
    with duckdb.connect(DB_PATH, read_only=True) as con:
        sensors = con.execute("SELECT sensor_index, latitude, longitude FROM sensor_table").df()

    spatial_weights = calculate_spatial_weights(sensors)
    weather_df = load_weather_df()

    end_date = datetime.now().strftime("%Y-%m-%d")
    data_all = get_data_all("2022-04-01", end_date)
    data_all = data_all.sort_index()

    # Apply EMA smoothing per sensor before modelling
    data_reset = data_all.reset_index()
    # Index name is lost when assigning date objects; restore it
    if "time_stamp" not in data_reset.columns:
        data_reset = data_reset.rename(columns={"index": "time_stamp"})
    data_reset = apply_ema_filter(data_reset, "pm25", span=7)
    data_reset["pm25"] = data_reset["pm25_ema"]
    data_all = data_reset.drop(columns=["pm25_ema"]).set_index("time_stamp")

    data_all = merge_weather(data_all, weather_df)

    # Vectorized spatial lag: pivot to (date × sensor), multiply once for all dates
    pm25_wide = data_all.pivot_table(
        index=data_all.index, columns="sensor_index", values="pm25", aggfunc="first"
    )
    sensor_order = spatial_weights.index.tolist()
    pm25_wide = pm25_wide.reindex(columns=sensor_order).fillna(0)
    spatial_lag_matrix = pm25_wide.values @ spatial_weights.values.T
    spatial_lag_df = pd.DataFrame(
        spatial_lag_matrix, index=pm25_wide.index, columns=sensor_order
    )

    split_date = pd.Timestamp(TEST_PERIOD_START).date()
    results = []

    for i, sensor_id in enumerate(sensor_order):
        print(f"Sensor {sensor_id} ({i + 1}/{len(sensor_order)})")

        sensor_data = data_all[data_all["sensor_index"] == sensor_id].copy()
        if sensor_data.empty:
            continue

        sensor_data["spatial_lag_pm25"] = (
            spatial_lag_df[sensor_id].reindex(sensor_data.index).values
        )

        feature_cols = [
            c for c in SPATIAL_FEATURES
            if c in sensor_data.columns and sensor_data[c].notna().any()
        ]
        if not feature_cols:
            continue

        train = sensor_data[sensor_data.index < split_date]
        test  = sensor_data[sensor_data.index >= split_date]

        if train.empty or test.empty:
            print(f"  Skipping: insufficient train/test data")
            continue

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("xgbreg", XGBRegressor(n_estimators=100, random_state=42)),
        ])
        pipeline.fit(train[feature_cols], train["pm25"])
        y_pred = pipeline.predict(test[feature_cols])

        for date, actual, pred in zip(test.index, test["pm25"].values, y_pred):
            results.append({
                "Date":         date,
                "Sensor Index": sensor_id,
                "Y Test":       actual,
                "Y Pred":       pred,
            })

    pd.DataFrame(results).to_csv("datasets/spatial_results.csv", index=False)
    print(f"Saved {len(results)} predictions to datasets/spatial_results.csv")
