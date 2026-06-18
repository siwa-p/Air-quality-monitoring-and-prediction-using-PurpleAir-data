import pandas as pd


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
