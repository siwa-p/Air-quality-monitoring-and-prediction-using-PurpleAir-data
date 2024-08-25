import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load and preprocess data
merged_data = pd.read_csv('../datasets/merged_data_CNN.csv', index_col=False)
merged_data['time_stamp'] = pd.to_datetime(merged_data['time_stamp'])
merged_data.set_index('time_stamp', inplace=True)
merged_data.sort_index(inplace=True)
merged_data.index = merged_data.index.to_period('D')


# Loop over all sensors and apply SARIMAX modeling
sensor_results = []
for sensor_index in merged_data['sensor_index'].unique():
    sensor_data = merged_data[merged_data['sensor_index'] == sensor_index]
    if sensor_data.empty:
        print(f"No data for sensor {sensor_index}")
        continue

    sensor_chosen = sensor_data['sensor_index'].iloc[0]
    latitude = sensor_data['latitude'].iloc[0]
    longitude = sensor_data['longitude'].iloc[0]
    train_data = sensor_data[:'2024-01-08']
    test_data = sensor_data['2024-01-08':]
    
    if train_data.empty or test_data.empty:
        print(f"Insufficient data for sensor {sensor_index}")
        continue

    y = train_data['pm2.5_atm_a']
    X = train_data.drop(columns=['pm2.5_atm_a', 'pm2.5_atm_b', 'pm2.5_cf_1_a', 'pm2.5_cf_1_b',
                                 'name', 'latitude', 'longitude', 'STATION', 'LATITUDE', 
                                 'LONGITUDE', 'ELEVATION'])
    X_test = test_data.drop(columns=['pm2.5_atm_a', 'pm2.5_atm_b', 'pm2.5_cf_1_a', 'pm2.5_cf_1_b',
                                     'name', 'latitude', 'longitude', 'STATION', 'LATITUDE', 
                                     'LONGITUDE', 'ELEVATION'])
    
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.fillna(X.mean())
    X_test = X_test.fillna(X_test.mean())
    
    if X.isnull().values.any() or X_test.isnull().values.any() or np.isinf(X).values.any() or np.isinf(X_test).values.any():
        print(f"NaNs or infs found in exogenous data for sensor {sensor_chosen}, skipping this sensor.")
        continue
    
    try:
        model = SARIMAX(y, exog=X, order=(1, 0, 1))
        model_fit = model.fit(maxiter=5000, method='bfgs')
    except Exception as e:
        print(f"Model fitting failed for sensor {sensor_chosen} due to {e}")
        continue

    try:
        predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, exog=X_test)
        if predictions.isna().any():
            predictions = np.nan

        sensor_results.append({
            'Sensor Index': sensor_chosen,
            'Latitude': latitude,
            'Longitude': longitude,
            'y_test': test_data['pm2.5_atm_a'].tolist(),
            'y_pred': predictions.tolist() if predictions is not np.nan else np.nan
        })
    except Exception as e:
        print(f"Prediction failed for sensor {sensor_chosen} due to {e}")
        continue

# Save results to CSV
sensor_results_df = pd.DataFrame(sensor_results)
sensor_results_df.to_csv('../datasets/sarimax_results.csv', index=False)
