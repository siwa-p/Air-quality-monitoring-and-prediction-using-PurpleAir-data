import numpy as np
import pandas as pd
import pickle
from skgstat import Variogram, OrdinaryKriging


def lat_lon_to_cartesian(lat, lon, radius=6371.0):
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return x, y, z


def create_spatial_map(data: pd.DataFrame, timestamp):
    sample = data[data["time_stamp"] == timestamp]
    lat = np.radians(sample["latitude"].values)
    lon = np.radians(sample["longitude"].values)
    values = sample["pm2_5_atm_a"].values
    cartesian = np.array([lat_lon_to_cartesian(la, lo) for la, lo in zip(lat, lon)])
    return cartesian[:, :2], values


def run_interpolation(
    input_csv: str = "datasets/merged_data.csv",
    output_values: str = "datasets/processed_data_values.pkl",
    output_errors: str = "datasets/processed_data_errors.pkl",
):
    data = pd.read_csv(input_csv)
    data = data[(data["pm2_5_atm_a"] < 50) & (data["pm2_5_atm_b"] < 50)]
    data = data.dropna(subset=["latitude", "longitude", "pm2_5_atm_a"])

    timestamps = data["time_stamp"].unique()
    processed_values = []
    processed_errors = []

    for ts in timestamps:
        spatial_map, values = create_spatial_map(data, ts)
        V = Variogram(spatial_map, values, maxlag=100, n_lags=20, use_nugget=True, max_nfev=5000)
        V.model = "spherical"

        try:
            ok = OrdinaryKriging(V, min_points=3, max_points=15, mode="exact")
            x, y = spatial_map[:, 0], spatial_map[:, 1]
            xx, yy = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
            field = ok.transform(xx.flatten(), yy.flatten()).reshape(xx.shape)
            sigma = ok.sigma.reshape(xx.shape)
            processed_values.append((field, xx, yy))
            processed_errors.append(sigma)
        except ZeroDivisionError:
            print(f"Kriging failed at timestamp {ts}: division by zero.")

    with open(output_values, "wb") as f:
        pickle.dump(processed_values, f)
    with open(output_errors, "wb") as f:
        pickle.dump(processed_errors, f)

    print(f"Saved {len(processed_values)} interpolated grids to {output_values}")


if __name__ == "__main__":
    run_interpolation()
