import duckdb
import pandas as pd
import numpy as np
from loguru import logger
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error

from src.config import DB_PATH, NOAA_STATIONS, ACTIVE_CITY, TEST_PERIOD_START, PM25_OUTLIER_THRESHOLD, SPATIAL_BBOX
from src.preprocessing.merge_data import load_weather_df
from src.preprocessing.sensor_filter import apply_hampel_filter

DROP_META = ["sensor_index", "name", "latitude", "longitude",
             "STATION", "LATITUDE", "LONGITUDE", "ELEVATION"]


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
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    sensor_data = sensor_data.copy()
    sensor_data["time_stamp"] = pd.to_datetime(sensor_data["time_stamp"])
    sensor_data = sensor_data.sort_values("time_stamp")

    # Hampel filter: remove sensor spikes. Target stays as raw (Hampel-cleaned) pm25.
    sensor_data["pm25"] = apply_hampel_filter(
        sensor_data["pm25"].reset_index(drop=True), window=3, k=3.0
    ).values
    # EMA span=3 applied only to lag/rolling features — not to the prediction target.
    sensor_data["pm25_smooth"] = sensor_data["pm25"].ewm(span=3, adjust=False).mean().values

    sensor_data.set_index("time_stamp", inplace=True)
    sensor_data.index = sensor_data.index.date

    weather_df = weather_df.copy()
    weather_df.index = pd.to_datetime(weather_df.index).date

    merged = pd.merge(sensor_data, weather_df, left_index=True, right_index=True)
    merged = merged[merged["pm25"] < PM25_OUTLIER_THRESHOLD]

    merged["pm25_lag1"]      = merged["pm25_smooth"].shift(1)
    merged["pm25_lag2"]      = merged["pm25_smooth"].shift(2)
    merged["pm25_lag7"]      = merged["pm25_smooth"].shift(7)
    merged["pm25_lag14"]     = merged["pm25_smooth"].shift(14)
    merged["pm25_lag30"]     = merged["pm25_smooth"].shift(30)
    merged["pm25_lag365"]    = merged["pm25_smooth"].shift(365)
    merged["pm25_roll7"]     = merged["pm25_smooth"].rolling(7, min_periods=1).mean()
    merged["pm25_roll7_std"] = merged["pm25_smooth"].rolling(7, min_periods=2).std()
    merged = merged.drop(columns=["pm25_smooth"])

    dt_index = pd.to_datetime(merged.index)
    merged["month"]       = dt_index.month
    merged["week_of_year"] = dt_index.isocalendar().week.values
    merged["day_of_week"] = dt_index.day_of_week

    merged.drop(columns=[c for c in DROP_META if c in merged.columns], inplace=True)
    return merged


def train_temporal_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    model = XGBRegressor()
    model.fit(X_train, y_train)
    return model


def evaluate_temporal_model(model: XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series):
    if len(X_test) == 0:
        return y_test, np.nan, np.nan
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    return y_test, y_pred, rmse


if __name__ == "__main__":
    logger.add("logs/temporal_run.log", rotation="10 MB", retention=3)
    weather_df = load_weather_df()

    bb = SPATIAL_BBOX
    with duckdb.connect(DB_PATH, read_only=True) as con:
        sensors = con.execute(f"""
            SELECT sensor_index, latitude, longitude FROM sensor_table
            WHERE longitude BETWEEN {bb['lon_min']} AND {bb['lon_max']}
              AND latitude  BETWEEN {bb['lat_min']} AND {bb['lat_max']}
        """).df()

    sensor_results = []
    all_importances: dict[int, dict] = {}
    for _, row in sensors.iterrows():
        sensor_chosen = int(row["sensor_index"])
        if not has_data_after_test_period(sensor_chosen, TEST_PERIOD_START):
            logger.debug(f"No data after test period for sensor {sensor_chosen}")
            continue

        data = load_data(sensor_chosen)
        processed = preprocess_temporal_data(data, weather_df)

        processed = processed.dropna(axis=1, how="all")
        if processed.empty or "pm25" not in processed.columns:
            logger.warning(f"No usable columns for sensor {sensor_chosen}")
            continue

        imputer = SimpleImputer()
        df_imp = pd.DataFrame(
            imputer.fit_transform(processed),
            columns=processed.columns,
            index=pd.to_datetime(processed.index),
        )

        train = df_imp[:TEST_PERIOD_START]
        test  = df_imp[TEST_PERIOD_START:]
        X_train = train.drop(columns=["pm25"])
        y_train = train["pm25"]
        X_test  = test.drop(columns=["pm25"])
        y_test  = test["pm25"]

        if len(X_train) == 0:
            logger.warning(f"No training samples for sensor {sensor_chosen}")
            continue

        model = train_temporal_model(X_train, y_train)
        y_test, y_pred, rmse = evaluate_temporal_model(model, X_test, y_test)

        if isinstance(y_pred, float):  # np.nan returned for empty test set
            continue

        all_importances[sensor_chosen] = dict(zip(X_train.columns, model.feature_importances_))
        logger.info(f"Sensor {sensor_chosen}: RMSE = {rmse:.2f} µg/m³ ({len(y_test)} test days)")
        for date, actual, pred in zip(y_test.index, y_test.values, y_pred):
            sensor_results.append({
                "Date":         date.date(),
                "Sensor Index": sensor_chosen,
                "Latitude":     row["latitude"],
                "Longitude":    row["longitude"],
                "Y Test":       actual,
                "Y Pred":       pred,
            })

    pd.DataFrame(sensor_results).to_csv("datasets/temporal_results.csv", index=False)
    logger.info(f"Saved {len(sensor_results)} rows to datasets/temporal_results.csv")

    if all_importances:
        imp_df = pd.DataFrame(all_importances).T.fillna(0)
        imp_df.mean().sort_values(ascending=False).to_csv(
            "datasets/temporal_feature_importance.csv", header=["importance"]
        )
        logger.info("Feature importance saved → datasets/temporal_feature_importance.csv")
