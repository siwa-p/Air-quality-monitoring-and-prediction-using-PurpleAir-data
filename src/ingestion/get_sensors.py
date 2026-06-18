import requests
import pandas as pd
import duckdb
from dotenv import load_dotenv
import os
from loguru import logger

from src.config import DB_PATH, BOUNDING_BOXES, ACTIVE_CITY

load_dotenv()
API_KEY = os.getenv("PURPLEAIR_API_KEY")
ROOT_URL = os.getenv("PURPLEAIR_API_URL")

FIELDS = ["sensor_index", "name", "latitude", "longitude"]
LOC_TYPE = {"indoor": "1", "outdoor": "0"}


def fetch_sensors_df(nwlng, nwlat, selng, selat, location="outdoor") -> pd.DataFrame:
    params = {
        "api_key": API_KEY,
        "fields": ",".join(FIELDS),
        "nwlng": nwlng,
        "nwlat": nwlat,
        "selng": selng,
        "selat": selat,
        **({"location_type": LOC_TYPE[location]} if location in LOC_TYPE else {}),
    }
    endpoint = f"{ROOT_URL}/sensors"
    url = requests.Request("GET", endpoint, params=params).prepare().url
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return pd.DataFrame.from_records(response.json().get("data", []), columns=FIELDS)
    except requests.RequestException:
        logger.exception(f"Failed to fetch sensors from {url}")
        raise


def get_sensors_list(nwlng, nwlat, selng, selat, location="outdoor") -> pd.DataFrame:
    sensors = fetch_sensors_df(nwlng, nwlat, selng, selat, location)
    with duckdb.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS sensor_table (
                sensor_index INTEGER PRIMARY KEY,
                name         TEXT,
                latitude     DOUBLE,
                longitude    DOUBLE
            )
        """)
        con.execute("INSERT OR IGNORE INTO sensor_table SELECT * FROM sensors")
        con.execute("CREATE INDEX IF NOT EXISTS idx_sensor ON sensor_table(sensor_index)")
    return sensors


def main():
    bbox = BOUNDING_BOXES[ACTIVE_CITY]
    sensors = get_sensors_list(**bbox, location="outdoor")
    print(sensors)


if __name__ == "__main__":
    main()
