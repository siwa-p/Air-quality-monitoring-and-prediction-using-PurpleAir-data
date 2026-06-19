import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors


def build_sensor_graph(sensors_df: pd.DataFrame, k: int = 5):
    """
    Build an undirected k-NN sensor graph from lat/lon coordinates.

    Returns:
        edge_index: LongTensor [2, E] — bidirectional edges
        edge_weight: FloatTensor [E]  — inverse-distance weights (1/km)
    """
    coords = sensors_df[["latitude", "longitude"]].values
    n = len(coords)
    k = min(k, n - 1)

    # ball_tree + haversine gives accurate great-circle distances on raw lat/lon;
    # kd_tree + Euclidean distorts East-West distances by ~25% at mid-latitudes.
    coords_rad = np.deg2rad(coords)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree", metric="haversine")
    nn.fit(coords_rad)
    distances, neighbors = nn.kneighbors(coords_rad)

    # Exclude self (index 0); convert radians → km
    distances_km = distances[:, 1:] * 6371.0
    neighbors    = neighbors[:, 1:]

    src, dst, weights = [], [], []
    for i in range(n):
        for j in range(k):
            d = max(float(distances_km[i, j]), 1e-3)  # clamp: never drop an edge
            src.append(i)
            dst.append(int(neighbors[i, j]))
            weights.append(1.0 / d)

    # Bidirectional edges — spatial sensor graphs are undirected
    edge_index  = torch.tensor([src + dst, dst + src], dtype=torch.long)
    edge_weight = torch.tensor(weights + weights, dtype=torch.float)
    return edge_index, edge_weight
