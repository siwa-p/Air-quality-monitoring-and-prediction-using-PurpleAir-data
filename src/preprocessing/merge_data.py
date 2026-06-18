import duckdb
import pandas as pd

from src.config import DB_PATH, NOAA_STATIONS, ACTIVE_CITY, TEST_PERIOD_START, PM25_OUTLIER_THRESHOLD

WEATHER_COLS = [
    "DATE", "AWND", "DAPR", "MDPR", "PGTM", "PRCP", "SNOW", "SNWD",
    "TAVG", "TMAX", "TMIN", "WDF2", "WDF5", "WESD", "WESF",
    "WSF2", "WSF5", "WT01", "WT02", "WT03", "WT04", "WT05", "WT06", "WT08",
]


def load_weather(weather_csv: str, station_id: str) -> pd.DataFrame:
    """Load and clean NOAA weather data for one station; returns date-indexed DataFrame."""
    weather = pd.read_csv(weather_csv)
    w = weather[weather["STATION"] == station_id][["STATION"] + WEATHER_COLS].copy()
    w = w.iloc[:-2].fillna(0)
    w["DATE"] = pd.to_datetime(w["DATE"])
    w.set_index("DATE", inplace=True)
    return w


def build_merged_csv(
    weather_csv: str,
    output_csv: str = "datasets/merged_data.csv",
    start_date: str = "2022-04-01",
    end_date: str = None,
) -> pd.DataFrame:
    station_id = NOAA_STATIONS[ACTIVE_CITY]

    with duckdb.connect(DB_PATH, read_only=True) as con:
        data = con.execute("""
            SELECT
                s.sensor_index, s.name, s.latitude, s.longitude,
                d.time_stamp,
                d.humidity_a, d.temperature_a, d.pressure_a,
                d.pm2_5_atm_a, d.pm2_5_atm_b, d.pm2_5_cf_1_a, d.pm2_5_cf_1_b
            FROM sensor_table AS s
            JOIN raw.data_daily AS d ON s.sensor_index = d.sensor_index
        """).df()

    data["time_stamp"] = pd.to_datetime(data["time_stamp"])
    data.set_index("time_stamp", inplace=True)
    data = data.sort_index()
    data.index = data.index.date

    weather = load_weather(weather_csv, station_id)
    weather.index = weather.index.date

    merged = data.join(weather, how="left")
    merged = merged[(merged.index >= pd.to_datetime(start_date).date())]
    if end_date:
        merged = merged[merged.index <= pd.to_datetime(end_date).date()]

    merged = merged[merged["pm2_5_atm_a"] < PM25_OUTLIER_THRESHOLD]
    merged = merged.reset_index().rename(columns={"index": "time_stamp"})
    merged.to_csv(output_csv, index=False)
    print(f"Saved {len(merged)} rows to {output_csv}")
    return merged


if __name__ == "__main__":
    build_merged_csv(
        weather_csv=f"datasets/{ACTIVE_CITY}_stations_data.csv",
        output_csv="datasets/merged_data.csv",
        start_date="2022-04-01",
    )
