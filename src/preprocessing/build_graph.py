import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors


def build_sensor_graph(sensors_df: pd.DataFrame, k: int = 5):
    """
    Build a k-NN sensor graph from lat/lon coordinates.

    Returns:
        edge_index: LongTensor [2, E] — source/destination node indices
        edge_weight: FloatTensor [E]  — inverse-distance weights (positive)
    """
    coords = sensors_df[["latitude", "longitude"]].values
    n = len(coords)
    k = min(k, n - 1)

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree")
    nn.fit(coords)
    distances, neighbors = nn.kneighbors(coords)

    # Exclude self (index 0)
    distances = distances[:, 1:]
    neighbors = neighbors[:, 1:]

    src, dst, weights = [], [], []
    for i in range(n):
        for j in range(k):
            d = distances[i, j]
            if d > 0:
                src.append(i)
                dst.append(neighbors[i, j])
                weights.append(1.0 / d)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)
    return edge_index, edge_weight
