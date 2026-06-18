import time
import requests
import pandas as pd
import duckdb
from io import StringIO
from datetime import datetime, timezone
from dotenv import load_dotenv
import os
from loguru import logger

MAX_RETRIES = 4
BACKOFF_BASE = 2  # seconds; doubles each retry: 2, 4, 8, 16
EARLY_EXIT_EMPTY = 3  # consecutive empty 30-day chunks → sensor predates that period

from src.config import DB_PATH

load_dotenv()
API_KEY = os.getenv("PURPLEAIR_API_KEY")
ROOT_URL = os.getenv("PURPLEAIR_API_URL")

FIELDS = [
    'pm2.5_atm_a', 'pm2.5_atm_b', 'pm2.5_cf_1_a', 'pm2.5_cf_1_b',
    'humidity_a', 'humidity_b', 'temperature_a', 'temperature_b',
    'pressure_a', 'pressure_b'
]
# Map API dot-names to table column names
FIELD_RENAME = {f: f.replace('.', '_') for f in FIELDS}

SLEEP_SECONDS = 3

TABLE_NAMES = {
    1440: 'raw.data_daily',
    60:   'raw.data_hourly',
}

def create_table_sql(table_name):
    return f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            time_stamp      TIMESTAMP,
            sensor_index    INTEGER,
            pm2_5_atm_a     DOUBLE,
            pm2_5_atm_b     DOUBLE,
            pm2_5_cf_1_a    DOUBLE,
            pm2_5_cf_1_b    DOUBLE,
            humidity_a      DOUBLE,
            humidity_b      DOUBLE,
            temperature_a   DOUBLE,
            temperature_b   DOUBLE,
            pressure_a      DOUBLE,
            pressure_b      DOUBLE,
            date_added      TIMESTAMP,
            PRIMARY KEY (sensor_index, time_stamp)
        )
    '''


def get_sensor_indices(con):
    return con.execute('SELECT sensor_index FROM sensor_table').df()['sensor_index'].tolist()


def build_date_list(bdate, edate, average_time):
    begindate = datetime.fromisoformat(bdate)
    enddate = datetime.fromisoformat(edate)
    freq = '2D' if average_time == 60 else '30D'
    datelist = pd.date_range(begindate, enddate, freq=freq).tolist()
    datelist.reverse()
    return [dt.strftime('%Y-%m-%dT%H:%M:%SZ') for dt in datelist]


def fetch_chunk(sensor, start, end, average_time):
    # Timestamps and fields must not be URL-encoded — build URL directly
    url = (
        f"{ROOT_URL}/sensors/{sensor}/history/csv"
        f"?api_key={API_KEY}"
        f"&average={average_time}"
        f"&start_timestamp={start}"
        f"&end_timestamp={end}"
        f"&fields={','.join(FIELDS)}"
    )
    for attempt in range(MAX_RETRIES):
        response = requests.get(url)
        if response.status_code == 429:
            wait = BACKOFF_BASE ** (attempt + 1)
            logger.warning(f"Rate limited (sensor {sensor}); retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)
            continue
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text))
    raise requests.HTTPError(f"Exceeded {MAX_RETRIES} retries for sensor {sensor} [{start} → {end}]")


def get_historical_data(bdate, edate, average_time):
    table = TABLE_NAMES.get(average_time)
    if table is None:
        raise ValueError(f"Unsupported average_time={average_time}. Must be one of {list(TABLE_NAMES)}")

    date_list = build_date_list(bdate, edate, average_time)
    len_datelist = len(date_list) - 1

    with duckdb.connect(DB_PATH) as con:
        con.execute('CREATE SCHEMA IF NOT EXISTS raw')
        con.execute(create_table_sql(table))
        con.execute(f'CREATE INDEX IF NOT EXISTS idx_sensor ON {table}(sensor_index)')

        sensors = get_sensor_indices(con)
        for sensor in sensors:
            consecutive_empty = 0
            for i in range(len_datelist):
                start, end = date_list[i + 1], date_list[i]
                time.sleep(SLEEP_SECONDS)

                try:
                    df = fetch_chunk(sensor, start, end, average_time)
                except requests.RequestException:
                    logger.exception(f"Failed to fetch sensor {sensor} for {start} to {end}")
                    continue

                if df.empty:
                    consecutive_empty += 1
                    logger.info(f"No data for sensor {sensor} from {start} to {end} ({consecutive_empty} consecutive empty)")
                    if consecutive_empty >= EARLY_EXIT_EMPTY:
                        logger.info(f"Sensor {sensor}: skipping remainder — no data for {consecutive_empty} consecutive chunks")
                        break
                    continue

                consecutive_empty = 0

                df = (
                    df
                    .drop_duplicates(keep='first')
                    .rename(columns=FIELD_RENAME)
                )
                df['sensor_index'] = sensor
                df['date_added'] = datetime.now(timezone.utc)

                df = df[['time_stamp', 'sensor_index',
                          'pm2_5_atm_a', 'pm2_5_atm_b',
                          'pm2_5_cf_1_a', 'pm2_5_cf_1_b',
                          'humidity_a', 'humidity_b',
                          'temperature_a', 'temperature_b',
                          'pressure_a', 'pressure_b',
                          'date_added']]

                con.execute(f'INSERT OR REPLACE INTO {table} SELECT * FROM df')
                logger.info(f"Upserted {len(df)} rows for sensor {sensor} from {start} to {end}")


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")

def main_daily():
    get_historical_data(
        bdate='2022-04-01T00:00:00+00:00',
        edate=_today_utc(),
        average_time=1440,
    )

def main_hourly():
    get_historical_data(
        bdate='2022-04-01T00:00:00+00:00',
        edate=_today_utc(),
        average_time=60,
    )

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'hourly':
        main_hourly()
    else:
        main_daily()
