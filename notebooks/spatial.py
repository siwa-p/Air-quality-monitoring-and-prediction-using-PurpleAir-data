import sqlite3
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta

def calculate_spatial_weights(sensors):
    coordinates = np.column_stack((sensors['latitude'], sensors['longitude']))
    nn_model = NearestNeighbors(n_neighbors=5+1, algorithm='kd_tree')
    nn_model.fit(coordinates)

    distances, neighbors = nn_model.kneighbors(coordinates)
    neighbors = neighbors[:, 1:]

    sensors= sensors.set_index('sensor_index')

    sensors['nearest_neighbors'] = [sensors.index[neighbors[i]].to_list() for i in range(len(sensors))]
    distances = distances[:,1:]
    weights = 1/distances
    weights_dict = {}
    for i, sensor_idx in enumerate(sensors.index):
        neighbor_indices = sensors.index[neighbors[i]]
        weights_dict[sensor_idx] = pd.Series(weights[i], index=neighbor_indices)
        
    spatial_weights = pd.DataFrame(weights_dict).fillna(0).T
    spatial_weights = spatial_weights.reindex(index=sensors.index, columns=sensors.index, fill_value=0)
    return spatial_weights

# method to get train test split for each sensor
def get_train_test_data_for_sensor(data, sensor_index, spatial_weights):
    # data.set_index('sensor_index', inplace=True)
    data['pm2.5_atm_a'].fillna(0, inplace=True)
    data['spatial_lag_pm2.5'] = spatial_weights.values @ data['pm2.5_atm_a'].values
    train_data = data[data.index != sensor_index]
    test_data = data[data.index == sensor_index]
    
    X_train = train_data[['humidity_a', 'temperature_a', 'pressure_a', 'spatial_lag_pm2.5']]
    y_train = train_data['pm2.5_atm_a']
    X_test = test_data[['humidity_a', 'temperature_a', 'pressure_a', 'spatial_lag_pm2.5']]
    y_test = test_data['pm2.5_atm_a']
    
    return X_train, X_test, y_train, y_test

def get_data_all(start_date, end_date):
    # Define the SQL query with the date range
    query = f"""
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
    WHERE 
        d.time_stamp BETWEEN '{start_date}T00:00:00Z' AND '{end_date}T23:59:59Z'
    """

    # Fetch data from the SQLite database
    with sqlite3.connect('datasets/dallas.sqlite') as db:
        data_d = pd.read_sql(query, db)

    data_d = data_d[data_d['pm2.5_atm_a'] < 1000]
    data_d['time_stamp'] = pd.to_datetime(data_d['time_stamp'])
    data_d.set_index('time_stamp', inplace=True)
    data_d.index = data_d.index.date
    data_d = data_d.sort_index(ascending=False)
    
    return data_d


def train_evaluate_xgboost_(X_train, X_test, y_train, y_test):
    # Define pipeline with SimpleImputer and XGBRegressor
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('xgbreg', XGBRegressor(n_estimators=100, random_state=42))
    ])

    # Train XGBoost model with pipeline
    pipeline.fit(X_train, y_train)

    # Predict PM2.5 values
    y_pred = pipeline.predict(X_test)
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    # r2 = r2_score(y_test, y_pred)
    
    return {
        'y_test': y_test,
        'mae': mae,
        'rmse': rmse,
        # 'r2': r2,
        'y_pred': y_pred
    }



query = """
    SELECT * FROM sensor_table
    """
with sqlite3.connect('datasets/dallas.sqlite') as db:
    sensors = pd.read_sql(query, db)

start_date = '2024-01-08'
end_date = '2024-03-29'
# Assuming spatial_weights is a predefined spatial weights matrix
spatial_weights = calculate_spatial_weights(sensors)  # Example placeholder
data_d = get_data_all(start_date, end_date)


results = []
current_date = datetime.strptime(start_date, '%Y-%m-%d')
end_date = datetime.strptime(end_date, '%Y-%m-%d')

while current_date <= end_date:
    date_data = data_d[data_d.index == current_date.date()]
    date_data = date_data[
        ~(date_data['pm2.5_atm_a'] > date_data['pm2.5_atm_b'] + 10) &
        ~(date_data['pm2.5_atm_a'] < date_data['pm2.5_atm_b'] - 10)
    ]
    
    date_data = sensors[['sensor_index']].merge(date_data, on='sensor_index', how='left')
    date_data.set_index('sensor_index', inplace=True)
    date_data = date_data[~date_data.index.duplicated(keep='first')]
    for sensor_index in date_data.index.unique():
        sensor_data = date_data[date_data.index == sensor_index]
        if not sensor_data.empty:
            X_train, X_test, y_train, y_test = get_train_test_data_for_sensor(date_data, sensor_index, spatial_weights)
            if not X_train.empty and not X_test.empty:
                y_pred = train_evaluate_xgboost_(X_train, X_test, y_train, y_test)
                results.append({
                    'Date': current_date.date(),
                    'Sensor Index': sensor_index,
                    'Y Test': y_pred['y_test'].values[0] if not y_pred['y_test'].empty else None,
                    'Y Pred': y_pred['y_pred'][0] if len(y_pred['y_pred']) > 0 else None,
                    'MAE': y_pred['mae'],
                    'RMSE': y_pred['rmse'],
                    # 'R2': y_pred['r2']
                })
    current_date += timedelta(days=1)

spatial_results = pd.DataFrame(results)
spatial_results.to_csv('datasets/spatial_results.csv', index=False)

