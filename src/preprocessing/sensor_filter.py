import pandas as pd


def apply_hampel_filter(
    series: pd.Series,
    window: int = 3,
    k: float = 3.0,
) -> pd.Series:
    """Replace statistical outliers with the local median.

    Outlier criterion: |x[i] - median(neighborhood)| > k * MAD(neighborhood)
    where neighborhood = [i-window, i+window]. Non-destructive for real events;
    only fires when a value is a genuine statistical anomaly relative to its neighbors.
    """
    s = series.copy().reset_index(drop=True).astype(float)
    n = len(s)
    for i in range(n):
        if pd.isna(s.iloc[i]):
            continue
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        neighborhood = s.iloc[lo:hi].dropna()
        if len(neighborhood) < 2:
            continue
        med = neighborhood.median()
        mad = (neighborhood - med).abs().median()
        if mad == 0:
            continue
        if abs(s.iloc[i] - med) > k * mad:
            s.iloc[i] = med
    return s


def apply_ema_filter(
    df: pd.DataFrame,
    column: str,
    span: int = 7,
    group_col: str = "sensor_index",
) -> pd.DataFrame:
    """Apply per-sensor EMA smoothing; adds `{column}_ema` column."""
    df = df.sort_values([group_col, "time_stamp"])
    df[f"{column}_ema"] = (
        df.groupby(group_col)[column]
        .transform(lambda s: s.ewm(span=span, adjust=False).mean())
    )
    return df
