import numpy as np
import pandas as pd
import pickle
from skgstat import Variogram, OrdinaryKriging

def lat_lon_to_cartesian(lat, lon, radius=6371.0):
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return x, y, z

def create_spatial_map(data, timestamp):
    sample_data = data[data['time_stamp'] == timestamp]
    lat = np.radians(sample_data['latitude'].values)
    lon = np.radians(sample_data['longitude'].values)
    values = sample_data['pm2.5_atm_a'].values

    # Convert latitude and longitude to Cartesian coordinates
    cartesian_points = np.array([lat_lon_to_cartesian(lat_i, lon_i) for lat_i, lon_i in zip(lat, lon)])
    cartesian_points = cartesian_points.reshape(-1, 3)    
    return cartesian_points[:, :2], values

# Load data
data = pd.read_csv('../datasets/merged_data.csv')
data = data[(data['pm2.5_atm_a'] < 50) & (data['pm2.5_atm_b'] < 50)]
data = data.dropna(subset=['latitude', 'longitude', 'pm2.5_atm_a'])

# Specify timestamps
timestamp = data['time_stamp'].unique()
# timestamp = ['2022-04-05', '2022-04-06', '2022-04-07', '2022-04-08']
# choose the last 100 timestamps
# timestamp = timestamp[-100:]

processed_data_values = []
processed_data_errors = []

for ts in timestamp:
    spatial_map, values = create_spatial_map(data, ts)
    V = Variogram(spatial_map, values, maxlag=100, n_lags=20, use_nugget=True, max_nfev=5000)
    V.model = 'spherical'
    # print(spatial_map, values)
    # print(V)
    
    try:
        ok = OrdinaryKriging(V, min_points=3, max_points=15, mode='exact')
        
        # Build the target grid
        x = spatial_map[:, 0]
        y = spatial_map[:, 1]
        xx, yy = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
        
        # Perform kriging interpolation
        field = ok.transform(xx.flatten(), yy.flatten()).reshape(xx.shape)
        sigma = ok.sigma.reshape(xx.shape)
        
        # Append results
        processed_data_values.append((field, xx, yy))
        processed_data_errors.append(sigma)
        
    except ZeroDivisionError:
        print(f'Error processing timestamp {ts}: Division by zero encountered in kriging.')

# processed_data_values now holds the interpolated fields and grid coordinates for each timestamp
# processed_data_errors holds the associated kriging errors


with open('../datasets/processed_data_values.pkl', 'wb') as f:
    pickle.dump(processed_data_values, f)
    
with open('../datasets/processed_data_errors.pkl', 'wb') as f:
    pickle.dump(processed_data_errors, f)

