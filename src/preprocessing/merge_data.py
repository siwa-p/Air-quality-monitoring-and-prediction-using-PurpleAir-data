import duckdb
import pandas as pd

from src.config import DB_PATH, NOAA_STATIONS, ACTIVE_CITY, PM25_OUTLIER_THRESHOLD
from src.preprocessing.sensor_filter import apply_ema_filter

# Columns where 0 is not a valid imputation — forward-fill instead
_FORWARD_FILL_COLS = ["TAVG", "TMAX", "TMIN", "AWND", "WDF2", "WDF5", "WSF2", "WSF5"]


def load_weather_df(station_id: str = None) -> pd.DataFrame:
    """Return weather data from DuckDB as a date-indexed DataFrame."""
    if station_id is None:
        station_id = NOAA_STATIONS[ACTIVE_CITY]
    with duckdb.connect(DB_PATH, read_only=True) as con:
        df = con.execute(
            "SELECT * FROM raw.weather_daily WHERE station = ?", [station_id]
        ).df()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").drop(columns=["station"]).fillna(0)
    return df


def build_merged_csv(
    output_csv: str = "datasets/merged_data.csv",
    start_date: str = "2022-04-01",
    end_date: str = None,
) -> pd.DataFrame:
    station_id = NOAA_STATIONS[ACTIVE_CITY]
    date_filter = f"AND d.time_stamp::DATE >= '{start_date}'"
    if end_date:
        date_filter += f" AND d.time_stamp::DATE <= '{end_date}'"

    query = f"""
        SELECT
            s.sensor_index, s.name, s.latitude, s.longitude,
            d.time_stamp,
            d.pm25,
            w.AWND, w.DAPR, w.MDPR, w.PGTM,
            w.PRCP, w.SNOW, w.SNWD,
            w.TAVG, w.TMAX, w.TMIN,
            w.WDF2, w.WDF5, w.WESD, w.WESF, w.WSF2, w.WSF5,
            w.WT01, w.WT02, w.WT03, w.WT04, w.WT05, w.WT06, w.WT08
        FROM sensor_table AS s
        JOIN raw.data_daily AS d ON s.sensor_index = d.sensor_index
        LEFT JOIN raw.weather_daily AS w
            ON d.time_stamp::DATE = w.date AND w.station = '{station_id}'
        WHERE d.pm25 < {PM25_OUTLIER_THRESHOLD}
        {date_filter}
        ORDER BY d.time_stamp, s.sensor_index
    """

    with duckdb.connect(DB_PATH, read_only=True) as con:
        merged = con.execute(query).df()

    merged["time_stamp"] = pd.to_datetime(merged["time_stamp"]).dt.date

    # Forward-fill temperature / wind before falling back to 0
    merged = merged.sort_values("time_stamp")
    for col in [c for c in _FORWARD_FILL_COLS if c in merged.columns]:
        merged[col] = merged[col].ffill()
    merged.fillna(0, inplace=True)

    # EMA smoothing per sensor to reduce measurement noise
    merged = merged.sort_values(["sensor_index", "time_stamp"])
    merged = apply_ema_filter(merged, "pm25", span=7)
    merged["pm25"] = merged["pm25_ema"]
    merged = merged.drop(columns=["pm25_ema"])

    merged.to_csv(output_csv, index=False)
    print(f"Saved {len(merged)} rows to {output_csv}")
    return merged


if __name__ == "__main__":
    build_merged_csv(output_csv="datasets/merged_data.csv", start_date="2022-04-01")
