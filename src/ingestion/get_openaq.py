import os
import time
import requests
import pandas as pd
import duckdb
from datetime import date, timedelta
from dotenv import load_dotenv
from loguru import logger

from src.config import DB_PATH, BOUNDING_BOXES, ACTIVE_CITY

load_dotenv()
TOKEN = os.getenv("OPENAQ_API_KEY")

BASE_URL = "https://api.openaq.org/v3"
PM25_PARAMETER_ID = 2
PAGE_LIMIT = 1000
SLEEP = 0.5
MAX_RETRIES = 4
BACKOFF_BASE = 2

SENSOR_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sensor_table (
    sensor_index INTEGER PRIMARY KEY,
    name         TEXT,
    latitude     DOUBLE,
    longitude    DOUBLE
)
"""

DATA_DAILY_SQL = """
CREATE TABLE IF NOT EXISTS raw.data_daily (
    time_stamp    TIMESTAMP,
    sensor_index  INTEGER,
    pm25          DOUBLE,
    PRIMARY KEY (sensor_index, time_stamp)
)
"""

DATA_HOURLY_SQL = """
CREATE TABLE IF NOT EXISTS raw.data_hourly (
    time_stamp    TIMESTAMP,
    sensor_index  INTEGER,
    pm25          DOUBLE,
    PRIMARY KEY (sensor_index, time_stamp)
)
"""


def _headers() -> dict:
    if not TOKEN:
        raise RuntimeError(
            "OPENAQ_API_KEY not set in .env — register free at https://explore.openaq.org/register"
        )
    return {"X-API-Key": TOKEN}


def _get(path: str, params: dict) -> dict:
    url = f"{BASE_URL}{path}"
    for attempt in range(MAX_RETRIES):
        resp = requests.get(url, headers=_headers(), params=params, timeout=60)
        if resp.status_code in (408, 429, 500):
            wait = BACKOFF_BASE ** (attempt + 1)
            logger.warning(f"HTTP {resp.status_code}; retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    raise requests.HTTPError(f"Exceeded {MAX_RETRIES} retries for {path}")


def discover_sensors(bbox: dict) -> list[dict]:
    """Return all PM2.5 sensors in the bounding box as {sensor_index, name, latitude, longitude}."""
    bbox_str = f"{bbox['nwlng']},{bbox['selat']},{bbox['selng']},{bbox['nwlat']}"
    sensors = []
    page = 1

    while True:
        data = _get("/locations", {
            "bbox": bbox_str,
            "parameters_id": PM25_PARAMETER_ID,
            "limit": PAGE_LIMIT,
            "page": page,
        })
        results = data.get("results", [])
        if not results:
            break

        for loc in results:
            coords = loc.get("coordinates") or {}
            lat = coords.get("latitude")
            lon = coords.get("longitude")
            if lat is None or lon is None:
                continue
            for s in loc.get("sensors", []):
                param = s.get("parameter") or {}
                if param.get("id") == PM25_PARAMETER_ID:
                    sensors.append({
                        "sensor_index": s["id"],
                        "name": loc.get("name", ""),
                        "latitude": lat,
                        "longitude": lon,
                    })
                    break

        logger.info(f"Page {page}: {len(results)} locations, {len(sensors)} PM2.5 sensors so far")
        if len(results) < PAGE_LIMIT:
            break
        page += 1
        time.sleep(SLEEP)

    return sensors


def _fetch_paged(path: str, date_from: date, date_to: date) -> list[dict]:
    records = []
    page = 1
    while True:
        data = _get(path, {
            "datetime_from": date_from.isoformat(),
            "datetime_to": date_to.isoformat(),
            "limit": PAGE_LIMIT,
            "page": page,
        })
        results = data.get("results", [])
        records.extend(results)
        if len(results) < PAGE_LIMIT:
            break
        page += 1
        time.sleep(SLEEP)
    return records


def _fetch_chunked(path: str, date_from: str, date_to: str) -> list[dict]:
    """Fetch in year-sized chunks to avoid server-side timeouts."""
    records = []
    cursor = date.fromisoformat(date_from)
    end = date.fromisoformat(date_to)
    while cursor <= end:
        chunk_end = min(date(cursor.year + 1, cursor.month, cursor.day) - timedelta(days=1), end)
        try:
            records.extend(_fetch_paged(path, cursor, chunk_end))
        except requests.HTTPError as e:
            logger.warning(f"Chunk {cursor} → {chunk_end} failed ({e}), skipping chunk")
        cursor = chunk_end + timedelta(days=1)
        time.sleep(SLEEP)
    return records


def _parse_rows(records: list[dict], sensor_id: int) -> list[dict]:
    rows = []
    for r in records:
        period = r.get("period") or {}
        dt_from = period.get("datetimeFrom") or {}
        ts = dt_from.get("utc")
        if ts is None or r.get("value") is None:
            continue
        rows.append({
            "time_stamp": pd.Timestamp(ts).tz_localize(None) if pd.Timestamp(ts).tzinfo is None else pd.Timestamp(ts).tz_convert("UTC").tz_localize(None),
            "sensor_index": sensor_id,
            "pm25": r["value"],
        })
    return rows


def ingest_daily(start_date: str, end_date: str, con: duckdb.DuckDBPyConnection):
    sensor_ids = con.execute("SELECT sensor_index FROM sensor_table").df()["sensor_index"].tolist()
    already_done = set(
        con.execute("SELECT DISTINCT sensor_index FROM raw.data_daily").df()["sensor_index"].tolist()
    )
    todo = [s for s in sensor_ids if s not in already_done]
    logger.info(
        f"Fetching daily data for {len(sensor_ids)} sensors ({start_date} → {end_date})"
        f" — {len(already_done)} already ingested, {len(todo)} remaining"
    )

    for sensor_id in todo:
        try:
            records = _fetch_chunked(f"/sensors/{sensor_id}/measurements/daily", start_date, end_date)
        except requests.HTTPError:
            logger.exception(f"Sensor {sensor_id}: skipping daily due to HTTP error")
            continue

        rows = _parse_rows(records, sensor_id)
        if not rows:
            logger.info(f"Sensor {sensor_id}: no daily data")
            continue

        df = pd.DataFrame(rows).drop_duplicates(subset=["sensor_index", "time_stamp"])
        con.execute("INSERT OR IGNORE INTO raw.data_daily SELECT * FROM df")
        logger.info(f"Sensor {sensor_id}: stored {len(df)} daily rows")
        time.sleep(SLEEP)


def ingest_hourly(start_date: str, end_date: str, con: duckdb.DuckDBPyConnection):
    sensor_ids = con.execute("SELECT sensor_index FROM sensor_table").df()["sensor_index"].tolist()
    already_done = set(
        con.execute("SELECT DISTINCT sensor_index FROM raw.data_hourly").df()["sensor_index"].tolist()
    )
    todo = [s for s in sensor_ids if s not in already_done]
    logger.info(
        f"Fetching hourly data for {len(sensor_ids)} sensors ({start_date} → {end_date})"
        f" — {len(already_done)} already ingested, {len(todo)} remaining"
    )

    for sensor_id in todo:
        try:
            records = _fetch_chunked(f"/sensors/{sensor_id}/hours", start_date, end_date)
        except requests.HTTPError:
            logger.exception(f"Sensor {sensor_id}: skipping hourly due to HTTP error")
            continue

        rows = _parse_rows(records, sensor_id)
        if not rows:
            logger.info(f"Sensor {sensor_id}: no hourly data")
            continue

        df = pd.DataFrame(rows).drop_duplicates(subset=["sensor_index", "time_stamp"])
        con.execute("INSERT OR IGNORE INTO raw.data_hourly SELECT * FROM df")
        logger.info(f"Sensor {sensor_id}: stored {len(df)} hourly rows")
        time.sleep(SLEEP)


def main():
    bbox = BOUNDING_BOXES[ACTIVE_CITY]
    daily_start = "2022-04-01"
    hourly_start = "2023-01-01"
    end_date = date.today().isoformat()

    logger.info(f"OpenAQ ingestion for {ACTIVE_CITY}")

    sensors = discover_sensors(bbox)
    if not sensors:
        logger.error("No PM2.5 sensors found in bounding box — check ACTIVE_CITY and bbox config")
        return

    with duckdb.connect(DB_PATH) as con:
        con.execute("CREATE SCHEMA IF NOT EXISTS raw")
        con.execute(SENSOR_TABLE_SQL)
        con.execute(DATA_DAILY_SQL)
        con.execute(DATA_HOURLY_SQL)

        df_sensors = pd.DataFrame(sensors)
        con.execute("INSERT OR REPLACE INTO sensor_table SELECT * FROM df_sensors")
        logger.info(f"Stored {len(sensors)} sensors")

        ingest_daily(daily_start, end_date, con)
        ingest_hourly(hourly_start, end_date, con)

    logger.info("OpenAQ ingestion complete")


if __name__ == "__main__":
    logger.add("logs/openaq_ingest.log", rotation="10 MB", retention=3)
    main()
