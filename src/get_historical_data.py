import time
import requests
import pandas as pd
import duckdb
from io import StringIO
from datetime import datetime, timezone
from dotenv import load_dotenv
import os
from loguru import logger

load_dotenv()
DB_PATH = "datasets/warehouse.duckdb"
API_KEY = os.getenv("PURPLEAIR-API")
ROOT_URL = os.getenv("PURPLEAIR-URL")

FIELDS = [
    'pm2.5_atm_a', 'pm2.5_atm_b', 'pm2.5_cf_1_a', 'pm2.5_cf_1_b',
    'humidity_a', 'humidity_b', 'temperature_a', 'temperature_b',
    'pressure_a', 'pressure_b'
]
# Map API dot-names to table column names
FIELD_RENAME = {f: f.replace('.', '_') for f in FIELDS}

SLEEP_SECONDS = 3

CREATE_TABLE_SQL = '''
    CREATE TABLE IF NOT EXISTS raw.data_table (
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
    freq = '14d' if average_time == 60 else '2d'
    datelist = pd.date_range(begindate, enddate, freq=freq).tolist()
    datelist.reverse()
    return [dt.strftime('%Y-%m-%dT%H:%M:%SZ') for dt in datelist]


def fetch_chunk(sensor, start, end, average_time):
    # Timestamps and fields must not be URL-encoded — build URL directly
    url = (
        f"{ROOT_URL}{sensor}/history/csv"
        f"?api_key={API_KEY}"
        f"&average={average_time}"
        f"&start_timestamp={start}"
        f"&end_timestamp={end}"
        f"&fields={','.join(FIELDS)}"
    )
    response = requests.get(url)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


def get_historical_data(bdate, edate, average_time):
    date_list = build_date_list(bdate, edate, average_time)
    len_datelist = len(date_list) - 1

    with duckdb.connect(DB_PATH) as con:
        con.execute('CREATE SCHEMA IF NOT EXISTS raw')
        con.execute(CREATE_TABLE_SQL)
        con.execute('CREATE INDEX IF NOT EXISTS idx_sensor ON raw.data_table(sensor_index)')

        sensors = get_sensor_indices(con)
        for sensor in sensors:
            for i in range(len_datelist):
                start, end = date_list[i + 1], date_list[i]
                time.sleep(SLEEP_SECONDS)

                try:
                    df = fetch_chunk(sensor, start, end, average_time)
                except requests.RequestException:
                    logger.exception(f"Failed to fetch sensor {sensor} for {start} to {end}")
                    continue

                if df.empty:
                    logger.info(f"No data for sensor {sensor} from {start} to {end}")
                    continue

                df = (
                    df
                    .drop_duplicates(keep='first')
                    .rename(columns=FIELD_RENAME)
                )
                df['sensor_index'] = sensor
                df['date_added'] = datetime.now(timezone.utc)

                # Reorder columns to match table schema
                df = df[['time_stamp', 'sensor_index',
                          'pm2_5_atm_a', 'pm2_5_atm_b',
                          'pm2_5_cf_1_a', 'pm2_5_cf_1_b',
                          'humidity_a', 'humidity_b',
                          'temperature_a', 'temperature_b',
                          'pressure_a', 'pressure_b',
                          'date_added']]

                con.execute('''
                    INSERT OR REPLACE INTO raw.data_table
                    SELECT * FROM df
                ''')
                logger.info(f"Upserted {len(df)} rows for sensor {sensor} from {start} to {end}")


def main():
    get_historical_data(
        bdate='2022-04-01T00:00:00+00:00',
        edate='2022-05-29T00:00:00+00:00',
        average_time=1440,
    )

if __name__ == "__main__":
    main()
