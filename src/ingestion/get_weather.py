import os
import time
import requests
import pandas as pd
import duckdb
from datetime import date
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from loguru import logger

from src.config import DB_PATH, NOAA_STATIONS, ACTIVE_CITY

load_dotenv()
TOKEN = os.getenv("NOAA_API_TOKEN")

BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"

DATATYPES = [
    "AWND", "DAPR", "MDPR", "PGTM",
    "PRCP", "SNOW", "SNWD",
    "TAVG", "TMAX", "TMIN",
    "WDF2", "WDF5", "WESD", "WESF", "WSF2", "WSF5",
    "WT01", "WT02", "WT03", "WT04", "WT05", "WT06", "WT08",
]

MAX_RESULTS = 1000
SLEEP = 0.5

CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS raw.weather_daily (
        date     DATE,
        station  TEXT,
        AWND     DOUBLE, DAPR  DOUBLE, MDPR  DOUBLE, PGTM  DOUBLE,
        PRCP     DOUBLE, SNOW  DOUBLE, SNWD  DOUBLE,
        TAVG     DOUBLE, TMAX  DOUBLE, TMIN  DOUBLE,
        WDF2     DOUBLE, WDF5  DOUBLE, WESD  DOUBLE, WESF  DOUBLE,
        WSF2     DOUBLE, WSF5  DOUBLE,
        WT01     DOUBLE, WT02  DOUBLE, WT03  DOUBLE, WT04  DOUBLE,
        WT05     DOUBLE, WT06  DOUBLE, WT08  DOUBLE,
        PRIMARY KEY (date, station)
    )
"""


def fetch_year(station_id: str, year_start: date, year_end: date) -> list[dict]:
    records = []
    offset = 1
    while True:
        params = {
            "datasetid":       "GHCND",
            "stationid":       f"GHCND:{station_id}",
            "startdate":       year_start.isoformat(),
            "enddate":         year_end.isoformat(),
            "datatypeid":      ",".join(DATATYPES),
            "units":           "standard",
            "limit":           MAX_RESULTS,
            "offset":          offset,
            "includemetadata": "false",
        }
        resp = requests.get(BASE_URL, headers={"token": TOKEN}, params=params, timeout=30)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        records.extend(results)
        logger.info(f"{year_start} → {year_end}: {len(records)} records")
        if len(results) < MAX_RESULTS:
            break
        offset += MAX_RESULTS
        time.sleep(SLEEP)
    return records


def fetch_and_store(station_id: str, start: str, end: str):
    start_dt = date.fromisoformat(start)
    end_dt   = date.fromisoformat(end)

    all_records = []
    cursor = start_dt
    while cursor <= end_dt:
        chunk_end = min(cursor + relativedelta(years=1) - relativedelta(days=1), end_dt)
        all_records.extend(fetch_year(station_id, cursor, chunk_end))
        cursor = chunk_end + relativedelta(days=1)
        time.sleep(SLEEP)

    if not all_records:
        raise RuntimeError(f"No data returned for station {station_id}")

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Pivot long → wide, one row per date
    wide = df.pivot_table(index="date", columns="datatype", values="value", aggfunc="first")
    wide = wide.reset_index()
    wide.columns.name = None
    wide.insert(1, "station", station_id)

    # Ensure all expected columns exist (fill missing datatypes with NaN)
    for col in DATATYPES:
        if col not in wide.columns:
            wide[col] = float("nan")
    wide = wide[["date", "station"] + DATATYPES]

    with duckdb.connect(DB_PATH) as con:
        con.execute("CREATE SCHEMA IF NOT EXISTS raw")
        con.execute(CREATE_TABLE_SQL)
        con.execute("INSERT OR REPLACE INTO raw.weather_daily SELECT * FROM wide")

    logger.info(f"Stored {len(wide)} rows for station {station_id} in raw.weather_daily")


def main():
    if not TOKEN:
        raise RuntimeError(
            "NOAA_API_TOKEN not set in .env — register free at https://www.ncdc.noaa.gov/cdo-web/token"
        )
    station_id = NOAA_STATIONS[ACTIVE_CITY]
    logger.info(f"Fetching NOAA data for {station_id} ({ACTIVE_CITY})")
    fetch_and_store(station_id, start="2022-04-01", end=date.today().isoformat())


if __name__ == "__main__":
    main()
