import pandas as pd
import numpy as np
import sqlite3

query = """
SELECT
    s.sensor_index,
    s.name,
    s.latitude,
    s.longitude,
    d.time_stamp,
    d.humidity_a,
    d.temperature_a,
    d.pressure_a,
    d."pm2.5_atm_a",
    d."pm2.5_atm_b",
    d."pm2.5_cf_1_a",
    d."pm2.5_cf_1_b"
FROM
    sensor_table AS s
JOIN
    data_table AS d
ON
    s.sensor_index = d.sensor_index

"""

with sqlite3.connect('../datasets/dallas.sqlite') as db:
    data = pd.read_sql(query, db)
    
data['time_stamp'] = pd.to_datetime(data['time_stamp']) # convert to datetime
data.set_index('time_stamp', inplace=True)  # set index to time_stamp
data =  data.sort_index(ascending=True) # sort the data

data.index = data.index.date # change index to date format
# Merging weather data from noaa to the data from a single sensor
weather_noaa_data = pd.read_csv('../datasets/Dallas_stations_data.csv')

query = """
SELECT
    s.sensor_index,
    s.name,
    s.latitude,
    s.longitude,
    d.time_stamp,
    d.humidity_a,
    d.temperature_a,
    d.pressure_a,
    d."pm2.5_atm_a",
    d."pm2.5_atm_b",
    d."pm2.5_cf_1_a",
    d."pm2.5_cf_1_b"
FROM
    sensor_table AS s
JOIN
    data_table AS d
ON
    s.sensor_index = d.sensor_index

"""

with sqlite3.connect('../datasets/dallas.sqlite') as db:
    data = pd.read_sql(query, db)
data['time_stamp'] = pd.to_datetime(data['time_stamp'])
data.set_index('time_stamp', inplace=True)
data =  data.sort_index(ascending=True)

data.index = data.index.date # change index to date format
# Merging weather data from noaa to the data from a single sensor
weather_noaa_data = pd.read_csv('../datasets/Dallas_stations_data.csv') # load the downloaded weather data from noaa

# filter the weather data for the station of interest
weather_noaa_data_w =  weather_noaa_data[weather_noaa_data['STATION']=='USW00003971'][['STATION', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'DATE', 'AWND', 'DAPR', 'MDPR', 'PGTM', 'PRCP', 'SNOW', 'SNWD', 'TAVG','TMAX','TMIN', 'WDF2','WDF5', 'WESD', 'WESF','WSF2', 'WSF5', 'WT01', 'WT02', 'WT03', 'WT04', 'WT05', 'WT06','WT08']]

# the noaa data is for two days longer
weather_noaa_data_w_n = weather_noaa_data_w[:-2] # the noaa data is for two days longer

# lot's of NAN values, which are essentially zeros
weather_noaa_data_w_n = weather_noaa_data_w_n.fillna(0)

# convert the date to datetime
weather_noaa_data_w_n['DATE'] = pd.to_datetime(weather_noaa_data_w_n['DATE'])

weather_noaa_data_w_n.set_index('DATE', inplace=True) # setting index makes it easier to merge later
merged_df = pd.merge(data, weather_noaa_data_w_n, how = 'left', left_index=True, right_index=True)
start_date = '2022-04-01'
end_date = '2024-03-29'

# Filter the DataFrame for rows within the specified range
filtered_merged_df = merged_df[(merged_df.index >= start_date) & (merged_df.index <= end_date)]

# Reset the index and rename the index column
filtered_merged_df = filtered_merged_df.reset_index().rename(columns={'index': 'time_stamp'})

filtered_merged_df = filtered_merged_df[filtered_merged_df['pm2.5_atm_a']<1000]

# Save the merged data to a CSV file
filtered_merged_df.to_csv('../datasets/merged_data.csv', index=False)