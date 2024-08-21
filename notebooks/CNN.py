import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.interpolate import griddata
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns


def lat_lon_to_cartesian(lat, lon, radius=6371.0):
    """Convert latitude and longitude to Cartesian coordinates."""
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return x, y, z

def create_spatial_map(data, timestamp, grid_x, grid_y):
    """Create a spatial map for a given timestamp."""
    subset = data[data['time_stamp'] == timestamp]
    lat = np.radians(subset['latitude'].values)
    lon = np.radians(subset['longitude'].values)
    values = subset['pm2.5_atm_a'].values

    cartesian_points = np.array([lat_lon_to_cartesian(lat_i, lon_i) for lat_i, lon_i in zip(lat, lon)]).reshape(-1, 3)
    grid_z = griddata(cartesian_points[:, :2], values, (grid_x, grid_y), method='cubic')
    grid_z_clipped = np.clip(grid_z, 0, None)
    
    return grid_z_clipped

def generate_spatial_maps(data):
    """Generate spatial maps for all timestamps."""
    min_lat, max_lat = np.radians(data['latitude'].min()), np.radians(data['latitude'].max())
    min_lon, max_lon = np.radians(data['longitude'].min()), np.radians(data['longitude'].max())

    min_x, min_y, _ = lat_lon_to_cartesian(min_lat, min_lon)
    max_x, max_y, _ = lat_lon_to_cartesian(max_lat, max_lon)

    x_res, y_res = (max_x - min_x) / 100, (max_y - min_y) / 100
    grid_x, grid_y = np.meshgrid(np.arange(min_x, max_x, x_res), np.arange(min_y, max_y, y_res))

    timestamps = data['time_stamp'].unique()
    spatial_maps = [create_spatial_map(data, ts, grid_x, grid_y) for ts in timestamps]

    spatial_maps = np.array(spatial_maps)
    # spatial_maps = spatial_maps / np.nanmax(spatial_maps)
    spatial_maps = np.nan_to_num(spatial_maps)
    
    return spatial_maps, grid_x, grid_y

def train_3dcnn_model(X_train, y_train):
    """Train a 3D CNN model."""
    model = Sequential([
        Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=(time_steps, 100, 101, 1), padding='same'),
        MaxPooling3D(pool_size=(2, 2, 2)),
        Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        MaxPooling3D(pool_size=(2, 2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(100*101*1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return predictions and errors."""
    loss = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    errors = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae
    }
    return y_test, y_pred, errors

def save_predictions(y_test, y_pred, grid_coordinates, sensor_cartesian_points, unique_sensor_locs, dates):
    all_data = pd.DataFrame()
    for idx, i in enumerate(range(0, len(y_test) )):
        if i >= len(y_pred):
            break
        predicted_sensor_values = griddata(
            (grid_coordinates[:, 0], grid_coordinates[:, 1]),
            y_pred[i],  
            sensor_cartesian_points[:, :2], 
            method='linear'
        )
        observed_sensor_values = griddata(
            (grid_coordinates[:, 0], grid_coordinates[:, 1]),
            y_test[i],  
            sensor_cartesian_points[:, :2], 
            method='linear'
        )
        values_list = []
        for j, (lat, lon) in enumerate(unique_sensor_locs):
            predicted_pm25 = predicted_sensor_values[j]
            observed_pm25 = observed_sensor_values[j]
            values_list.append({'Date': dates[i], 'Latitude': lat, 'Longitude': lon, 'Predicted_PM2.5': predicted_pm25, 'Observed_PM2.5': observed_pm25})

        pm_values_df = pd.DataFrame(values_list)
        pm_values_df['Predicted_PM2.5'] = pm_values_df['Predicted_PM2.5'].apply(lambda x: round(x, 5))
        pm_values_df['Observed_PM2.5'] = pm_values_df['Observed_PM2.5'].apply(lambda x: round(x, 5))
        pm_values_df['Absolute_Error'] = (pm_values_df['Predicted_PM2.5'] - pm_values_df['Observed_PM2.5']).abs()
        pm_values_df['Squared_Error'] = (pm_values_df['Predicted_PM2.5'] - pm_values_df['Observed_PM2.5']) ** 2
        pm_values_df['Percentage_Error'] = (pm_values_df['Absolute_Error'] / pm_values_df['Observed_PM2.5']).abs() * 100
        all_data = pd.concat([all_data, pm_values_df], ignore_index=True)
        
    all_data.to_csv('datasets/daily_predictions_CNN_not_normalized.csv' ,index=False)


if __name__ == "__main__":
    # Load preprocessed data
    filtered_merged_df = pd.read_csv('datasets/merged_data_CNN.csv', index_col=False)
    filtered_merged_df = filtered_merged_df[(filtered_merged_df['pm2.5_atm_a']<500) & (filtered_merged_df['pm2.5_atm_b']<500)& (filtered_merged_df['pm2.5_atm_a']>0)]

    # Generate spatial maps
    spatial_maps, grid_x, grid_y = generate_spatial_maps(filtered_merged_df)

    # Prepare data for 3D CNN
    time_steps = 10
    X, y = [], []
    for i in range(len(spatial_maps) - time_steps):
        X.append(spatial_maps[i:i + time_steps])
        y.append(spatial_maps[i + time_steps])

    X = np.array(X)[..., np.newaxis]
    y = np.array(y)[..., np.newaxis]
    y = y.reshape(y.shape[0], -1)

    # Train-test split
    split_index = int(0.9 * len(X))  
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Load model if already trained else train
    model_path = 'models/3dcnn_model.keras'

    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        # Train the model if it doesn't exist
        model = train_3dcnn_model(X_train, y_train)
        model.save(model_path)

    # Evaluate the model
    y_test, y_pred, errors = evaluate_model(model, X_test, y_test)

    # Prepare data for displaying plots
    unique_sensor_locs = filtered_merged_df[['latitude', 'longitude']].drop_duplicates().values
    sensor_cartesian_points = np.array([lat_lon_to_cartesian(np.radians(lat), np.radians(lon)) for lat, lon in unique_sensor_locs]).reshape(-1, 3)
    grid_coordinates = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # Define the initial date and the end date
    start_date = datetime.strptime('2022-04-01', '%Y-%m-%d')
    end_date = datetime.strptime('2024-03-29', '%Y-%m-%d')

    # Calculate the 648th day
    day_zero = start_date + timedelta(days=len(X_train))

    # Generate a list of dates from day_648 to end_date
    current_date = day_zero
    dates = []
    while current_date <= end_date:
        dates.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    # Display sensor plots
    # save_predictions(y_test, y_pred, grid_coordinates, sensor_cartesian_points, unique_sensor_locs, dates)

    # Reshape to the 2 dimensions
    y_test_reshaped = y_test.reshape(-1, 100, 101)
    y_pred_reshaped = y_pred.reshape(-1, 100, 101)

    # Calculate vmin and vmax for consistent color scaling
    vmin = min(y_test_reshaped.min(), y_pred_reshaped.min())
    vmax = max(y_test_reshaped.max(), y_pred_reshaped.max())

    step_size = 20

    # Display heatmaps of test data and predictions for a few timepoints.
    for i in range(0, len(y_test_reshaped), step_size):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        sns.heatmap(y_test_reshaped[i], cmap='viridis', cbar=True, vmin=vmin, vmax=vmax-50, ax=ax[0])
        ax[0].set_title(f'True PM2.5 Levels ({dates[i]})')
        sns.heatmap(y_pred_reshaped[i], cmap='viridis', cbar=True, vmin=vmin, vmax=vmax-50, ax=ax[1])
        ax[1].set_title(f'Predicted PM2.5 Levels ({dates[i]})')
        plt.show()
