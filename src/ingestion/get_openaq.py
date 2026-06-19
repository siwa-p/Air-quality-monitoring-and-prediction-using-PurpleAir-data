import os
import time
import requests
import pandas as pd
import duckdb
from datetime import date, timedelta
from dotenv import load_dotenv
from loguru import logger

from src.config import DB_PATH, BOUNDING_BOXES, ACTIVE_CITY, DAILY_START

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
        ts_parsed = pd.Timestamp(ts)
        rows.append({
            "time_stamp": ts_parsed.tz_localize(None) if ts_parsed.tzinfo is None
                          else ts_parsed.tz_convert("UTC").tz_localize(None),
            "sensor_index": sensor_id,
            "pm25": r["value"],
        })
    return rows


def ingest_daily(start_date: str, end_date: str, con: duckdb.DuckDBPyConnection):
    """Fetch daily data for sensors not yet in the warehouse."""
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


def backfill_daily(backfill_start: str, con: duckdb.DuckDBPyConnection):
    """Fill the gap from backfill_start up to the earliest date already in the warehouse.

    Skips sensors that already have data before the current warehouse start.
    Safe to re-run — INSERT OR IGNORE prevents duplicates.
    """
    sensor_ids = con.execute("SELECT sensor_index FROM sensor_table").df()["sensor_index"].tolist()

    # find the earliest timestamp currently in the warehouse
    row = con.execute("SELECT MIN(time_stamp) FROM raw.data_daily").fetchone()
    if not row or row[0] is None:
        logger.info("No existing daily data — nothing to backfill (run ingest_daily first)")
        return

    existing_start = pd.Timestamp(row[0]).date()
    backfill_end = (existing_start - timedelta(days=1)).isoformat()

    if backfill_start >= backfill_end:
        logger.info(f"Backfill window empty ({backfill_start} → {backfill_end}), nothing to do")
        return

    # skip sensors that already have data before the warehouse start
    already_backfilled = set(
        con.execute(
            f"SELECT DISTINCT sensor_index FROM raw.data_daily WHERE time_stamp < '{existing_start}'"
        ).df()["sensor_index"].tolist()
    )
    todo = [s for s in sensor_ids if s not in already_backfilled]
    logger.info(
        f"Backfill {backfill_start} → {backfill_end}: "
        f"{len(todo)} sensors to fetch, {len(already_backfilled)} already done"
    )

    for sensor_id in todo:
        try:
            records = _fetch_chunked(
                f"/sensors/{sensor_id}/measurements/daily", backfill_start, backfill_end
            )
        except requests.HTTPError:
            logger.exception(f"Sensor {sensor_id}: skipping backfill due to HTTP error")
            continue

        rows = _parse_rows(records, sensor_id)
        if not rows:
            logger.info(f"Sensor {sensor_id}: no data found in backfill window")
            continue

        df = pd.DataFrame(rows).drop_duplicates(subset=["sensor_index", "time_stamp"])
        con.execute("INSERT OR IGNORE INTO raw.data_daily SELECT * FROM df")
        logger.info(f"Sensor {sensor_id}: backfilled {len(df)} rows ({backfill_start} → {backfill_end})")
        time.sleep(SLEEP)


def main():
    bbox = BOUNDING_BOXES[ACTIVE_CITY]
    end_date = date.today().isoformat()

    logger.info(f"OpenAQ ingestion for {ACTIVE_CITY} ({DAILY_START} → {end_date})")

    sensors = discover_sensors(bbox)
    if not sensors:
        logger.error("No PM2.5 sensors found in bounding box — check ACTIVE_CITY and bbox config")
        return

    with duckdb.connect(DB_PATH) as con:
        con.execute("CREATE SCHEMA IF NOT EXISTS raw")
        con.execute(SENSOR_TABLE_SQL)
        con.execute(DATA_DAILY_SQL)

        df_sensors = pd.DataFrame(sensors)
        con.execute("INSERT OR REPLACE INTO sensor_table SELECT * FROM df_sensors")
        logger.info(f"Stored {len(sensors)} sensors")

        ingest_daily(DAILY_START, end_date, con)
        backfill_daily(DAILY_START, con)

    logger.info("OpenAQ ingestion complete")


if __name__ == "__main__":
    logger.add("logs/openaq_ingest.log", rotation="10 MB", retention=3)
    main()
