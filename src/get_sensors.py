import requests
import pandas as pd
import duckdb
from dotenv import load_dotenv
import os
from loguru import logger

load_dotenv()
DB_PATH = "datasets/warehouse.duckdb"
API_KEY = os.getenv("PURPLEAIR-API")
root_url = os.getenv("PURPLEAIR-URL")
fields_list = ['sensor_index', 'name', 'latitude', 'longitude']

def fetch_sensors_df(nwlng, nwlat, selng, selat, location):
    loc_type = {'indoor': '1', 'outdoor': '0'}
    params = {
        'api_key': API_KEY,
        'fields': ','.join(fields_list),
        'nwlng': nwlng, 
        'nwlat': nwlat,
        'selng': selng, 
        'selat': selat,
        **({'location_type': loc_type[location]} if location in loc_type else {}),
    }
    try:
        response = requests.get(root_url, params=params)
        response.raise_for_status()
        return pd.DataFrame.from_records(response.json().get('data', []), columns=fields_list)
    except requests.RequestException as e:
        logger.exception(f"Failed to fetch data from {response.url}")
        raise e


def get_sensors_list(nwlng, nwlat, selng, selat, location):
    sensors = fetch_sensors_df(nwlng, nwlat, selng, selat, location)
    with duckdb.connect(DB_PATH) as con:
        con.execute('CREATE TABLE IF NOT EXISTS sensor_table (sensor_index INTEGER PRIMARY KEY, name TEXT, latitude DOUBLE, longitude DOUBLE)')
        con.execute('INSERT OR IGNORE INTO sensor_table SELECT * FROM sensors')
        con.execute('CREATE INDEX IF NOT EXISTS sensor_index ON sensor_table(sensor_index)')
    return sensors

def main():
    # Bounding Box for Dallas
    # 33.308403956633406, -97.42744748978863
    # 32.36533339607889, -96.27246459337103
    nwlng = -97.43  # Northwest longitude of the bounding box
    nwlat = 33.31   # Northwest latitude of the bounding box
    selng = -96.28  # Southeast longitude of the bounding box
    selat = 32.37  # Southeast latitude of the bounding box
    location = 'outdoor'  # You can specify 'indoor', 'outdoor', or 'all'
    sensors_list = get_sensors_list(nwlng, nwlat, selng, selat, location)
    print(sensors_list)

if __name__=="__main__":
    main()