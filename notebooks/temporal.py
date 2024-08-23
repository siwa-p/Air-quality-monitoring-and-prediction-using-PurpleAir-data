import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

def load_data(sensor_chosen):
    with sqlite3.connect('datasets/dallas.sqlite') as db:
        query = f"""
        SELECT * FROM data_table
        WHERE sensor_index = {sensor_chosen}
        """
        data_chosen_sensor = pd.read_sql(query, db)
    return data_chosen_sensor

def has_data_after_test_period(sensor_chosen, test_period):
    with sqlite3.connect('datasets/dallas.sqlite') as db:
        query = f"""
        SELECT COUNT(*) FROM data_table
        WHERE sensor_index = {sensor_chosen}
        AND time_stamp > '{test_period}'
        """
        count = pd.read_sql(query, db).iloc[0, 0]
    return count > 0

def preprocess_temporal_data(sensor_data, weather_data_file, station_id):
    # Convert timestamp to datetime and set as index
    sensor_data['time_stamp'] = pd.to_datetime(sensor_data['time_stamp'])
    sensor_data.set_index('time_stamp', inplace=True)
    sensor_data = sensor_data.sort_index(ascending=True)
    sensor_data.index = sensor_data.index.date # change index to date format
    
    # Read weather data
    weather_noaa_data = pd.read_csv(weather_data_file)
    
    # Filter weather data for the chosen station and select relevant columns
    weather_noaa_data_w = weather_noaa_data[weather_noaa_data['STATION'] == station_id][['DATE', 'AWND', 'DAPR', 'MDPR', 'PGTM', 'PRCP', 'SNOW', 'SNWD', 'TAVG', 'TMAX', 'TMIN', 'WDF2', 'WDF5', 'WESD', 'WESF', 'WSF2', 'WSF5', 'WT01', 'WT02', 'WT03', 'WT04', 'WT05', 'WT06', 'WT08']]
    
    # Remove NaN values and set date as index
    weather_noaa_data_w_n = weather_noaa_data_w[:-2]
    weather_noaa_data_w_n = weather_noaa_data_w_n.fillna(0)
    weather_noaa_data_w_n['DATE'] = pd.to_datetime(weather_noaa_data_w_n['DATE'])
    weather_noaa_data_w_n.set_index('DATE', inplace=True)
    
    # Merge sensor data with weather data based on date index
    merged_df = pd.merge(sensor_data, weather_noaa_data_w_n, left_index=True, right_index=True)
    
    # Remove outliers
    merged_df = merged_df[merged_df['pm2.5_atm_a'] < 1000]
    
    # Create time lag feature
    merged_df['pm2.5_lag1'] = merged_df['pm2.5_atm_a'].shift(1)
    merged_df = merged_df.fillna(0)
    
    # Remove unwanted columns
    columns_to_keep = [col for col in merged_df.columns if not col.endswith('_b') and col not in ['pm2.5_cf_1_a', 'STATION', 'LATITUDE', 'LONGITUDE', 'ELEVATION']]
    merged_df = merged_df[columns_to_keep]
    
    return merged_df

def train_temporal_model(X_train, y_train):
    # Fit the temporal model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = XGBRegressor()
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def evaluate_temporal_model(model, scaler, X_test, y_test):
    if len(X_test) == 0:
        return y_test, np.nan, np.nan
    else:
        # Make predictions
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        return y_test, y_pred, mse

if __name__ == "__main__":
    sensor_results = []
    query = """
    SELECT sensor_index, latitude, longitude FROM sensor_table
    """
    with sqlite3.connect('datasets/dallas.sqlite') as db:
        sensors = pd.read_sql(query, db)
    
    test_period = '2024-01-08'  # test period starts from this date
        
    for _, row in sensors.iterrows():
        sensor_chosen = row['sensor_index']
        latitude = row['latitude']
        longitude = row['longitude']

        if has_data_after_test_period(sensor_chosen, test_period):
            data_chosen_sensor = load_data(sensor_chosen)
            preprocessed_data = preprocess_temporal_data(data_chosen_sensor, 'datasets/Dallas_stations_data.csv','USW00003971')

            # Perform imputation
            imputer = SimpleImputer()
            df_imputed = pd.DataFrame(imputer.fit_transform(preprocessed_data), columns=preprocessed_data.columns, index=preprocessed_data.index)

            # Split the data into training and testing sets
            train_data = df_imputed[:test_period]
            test_data = df_imputed[test_period:]

            # Separate features and target for both training and testing sets
            X_train = train_data.drop(columns=['pm2.5_atm_a'])
            y_train = train_data['pm2.5_atm_a']
            X_test = test_data.drop(columns=['pm2.5_atm_a'])
            y_test = test_data['pm2.5_atm_a']
            
            if len(X_train) == 0:
                print(f"No samples found in the training data for sensor {sensor_chosen}")
                continue
            
            model, scaler = train_temporal_model(X_train, y_train)
            
            y_test, y_pred, mse = evaluate_temporal_model(model, scaler, X_test, y_test)

            # Append results for this sensor index to the list
            sensor_results.append({
                'Sensor Index': sensor_chosen,
                'Latitude': latitude,
                'Longitude': longitude,
                'Mean Squared Error': mse,
                'y_test': y_test.tolist(),
                'y_pred': y_pred.tolist() if y_pred is not np.nan else np.nan
            })
        else:
            print(f"No data found for sensor {sensor_chosen} after the test period.")
    
    temporal_results = pd.DataFrame(sensor_results)
    temporal_results.to_csv('datasets/temporal_results.csv', index=False)
