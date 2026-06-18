import duckdb
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error

from src.config import DB_PATH, NOAA_STATIONS, ACTIVE_CITY, TEST_PERIOD_START, PM25_OUTLIER_THRESHOLD


def load_data(sensor_chosen: int) -> pd.DataFrame:
    with duckdb.connect(DB_PATH, read_only=True) as con:
        return con.execute(
            "SELECT * FROM raw.data_daily WHERE sensor_index = ?", [sensor_chosen]
        ).df()


def has_data_after_test_period(sensor_chosen: int, test_period: str) -> bool:
    with duckdb.connect(DB_PATH, read_only=True) as con:
        count = con.execute(
            "SELECT COUNT(*) FROM raw.data_daily WHERE sensor_index = ? AND time_stamp > ?",
            [sensor_chosen, test_period],
        ).fetchone()[0]
    return count > 0


def preprocess_temporal_data(
    sensor_data: pd.DataFrame,
    weather_data_file: str,
    station_id: str,
) -> pd.DataFrame:
    sensor_data["time_stamp"] = pd.to_datetime(sensor_data["time_stamp"])
    sensor_data.set_index("time_stamp", inplace=True)
    sensor_data = sensor_data.sort_index()
    sensor_data.index = sensor_data.index.date

    weather = pd.read_csv(weather_data_file)
    weather_station = weather[weather["STATION"] == station_id][
        ["DATE", "AWND", "DAPR", "MDPR", "PGTM", "PRCP", "SNOW", "SNWD",
         "TAVG", "TMAX", "TMIN", "WDF2", "WDF5", "WESD", "WESF", "WSF2",
         "WSF5", "WT01", "WT02", "WT03", "WT04", "WT05", "WT06", "WT08"]
    ].copy()
    weather_station = weather_station.iloc[:-2].fillna(0)
    weather_station["DATE"] = pd.to_datetime(weather_station["DATE"])
    weather_station.set_index("DATE", inplace=True)

    merged = pd.merge(sensor_data, weather_station, left_index=True, right_index=True)
    merged = merged[merged["pm2_5_atm_a"] < PM25_OUTLIER_THRESHOLD]

    # Lag features
    merged["pm2_5_lag1"] = merged["pm2_5_atm_a"].shift(1)
    merged["pm2_5_lag7"] = merged["pm2_5_atm_a"].shift(7)
    merged["pm2_5_lag14"] = merged["pm2_5_atm_a"].shift(14)
    merged["pm2_5_lag30"] = merged["pm2_5_atm_a"].shift(30)

    # Rolling mean
    merged["pm2_5_roll7"] = (
        merged["pm2_5_atm_a"].rolling(7, min_periods=1).mean()
    )

    # Calendar features — derive from the DatetimeIndex before .date conversion
    dt_index = pd.to_datetime(merged.index)
    merged["month"] = dt_index.month
    merged["week_of_year"] = dt_index.isocalendar().week.values
    merged["day_of_week"] = dt_index.day_of_week

    columns_to_drop = (
        [c for c in merged.columns if c.endswith("_b")]
        + ["pm2_5_cf_1_a", "STATION", "LATITUDE", "LONGITUDE", "ELEVATION"]
    )
    merged.drop(columns=[c for c in columns_to_drop if c in merged.columns], inplace=True)

    return merged


def train_temporal_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    model = XGBRegressor()
    model.fit(X_train, y_train)
    return model


def evaluate_temporal_model(
    model: XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series
):
    if len(X_test) == 0:
        return y_test, np.nan, np.nan
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    return y_test, y_pred, rmse


if __name__ == "__main__":
    station_id = NOAA_STATIONS[ACTIVE_CITY]

    with duckdb.connect(DB_PATH, read_only=True) as con:
        sensors = con.execute(
            "SELECT sensor_index, latitude, longitude FROM raw.sensor_table"
        ).df()

    sensor_results = []
    for _, row in sensors.iterrows():
        sensor_chosen = int(row["sensor_index"])
        if not has_data_after_test_period(sensor_chosen, TEST_PERIOD_START):
            print(f"No data after test period for sensor {sensor_chosen}")
            continue

        data = load_data(sensor_chosen)
        processed = preprocess_temporal_data(
            data, f"datasets/{ACTIVE_CITY}_stations_data.csv", station_id
        )

        imputer = SimpleImputer()
        df_imp = pd.DataFrame(
            imputer.fit_transform(processed),
            columns=processed.columns,
            index=processed.index,
        )

        train = df_imp[:TEST_PERIOD_START]
        test = df_imp[TEST_PERIOD_START:]
        X_train = train.drop(columns=["pm2_5_atm_a"])
        y_train = train["pm2_5_atm_a"]
        X_test = test.drop(columns=["pm2_5_atm_a"])
        y_test = test["pm2_5_atm_a"]

        if len(X_train) == 0:
            print(f"No training samples for sensor {sensor_chosen}")
            continue

        model = train_temporal_model(X_train, y_train)
        y_test, y_pred, rmse = evaluate_temporal_model(model, X_test, y_test)

        sensor_results.append({
            "Sensor Index": sensor_chosen,
            "Latitude": row["latitude"],
            "Longitude": row["longitude"],
            "RMSE": rmse,
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist() if not isinstance(y_pred, float) else np.nan,
        })

    pd.DataFrame(sensor_results).to_csv("datasets/temporal_results.csv", index=False)
