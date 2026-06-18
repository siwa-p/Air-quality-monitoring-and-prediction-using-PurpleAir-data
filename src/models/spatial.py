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
    data_d = get_data_all(TEST_PERIOD_START, end_date)
    data_d = merge_weather(data_d, weather_df)

    results = []
    current_date = datetime.strptime(TEST_PERIOD_START, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    while current_date <= end_dt:
        date_data = data_d[data_d.index == current_date.date()].copy()
        date_data = (
            sensors[["sensor_index"]]
            .merge(date_data.reset_index(), on="sensor_index", how="left")
            .set_index("sensor_index")
        )
        date_data = date_data[~date_data.index.duplicated(keep="first")]

        for sensor_index in date_data.index.unique():
            if date_data.loc[[sensor_index]].empty:
                continue
            X_train, X_test, y_train, y_test = get_train_test_data_for_sensor(
                date_data, sensor_index, spatial_weights
            )
            if X_train.empty or X_test.empty:
                continue
            out = train_evaluate_spatial(X_train, X_test, y_train, y_test)
            results.append({
                "Date":         current_date.date(),
                "Sensor Index": sensor_index,
                "Y Test":  out["y_test"].values[0] if len(out["y_test"]) > 0 else None,
                "Y Pred":  out["y_pred"][0]         if len(out["y_pred"]) > 0 else None,
                "MAE":  out["mae"],
                "RMSE": out["rmse"],
            })

        current_date += timedelta(days=1)

    pd.DataFrame(results).to_csv("datasets/spatial_results.csv", index=False)
