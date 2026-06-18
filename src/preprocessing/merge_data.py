import duckdb
import pandas as pd

from src.config import DB_PATH, NOAA_STATIONS, ACTIVE_CITY, PM25_OUTLIER_THRESHOLD


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
    merged.fillna({
        "AWND": 0, "PRCP": 0, "SNOW": 0, "SNWD": 0, "TAVG": 0,
        "TMAX": 0, "TMIN": 0, "WDF2": 0, "WDF5": 0, "WSF2": 0,
        "WSF5": 0, "WT01": 0, "WT02": 0, "WT03": 0, "WT04": 0,
        "WT05": 0, "WT06": 0, "WT08": 0,
    }, inplace=True)

    merged.to_csv(output_csv, index=False)
    print(f"Saved {len(merged)} rows to {output_csv}")
    return merged


if __name__ == "__main__":
    build_merged_csv(output_csv="datasets/merged_data.csv", start_date="2022-04-01")
