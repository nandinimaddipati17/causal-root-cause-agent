"""
anomaly_detection.py
---------------------
The agent's "eyes": detects WHERE in a time series something changed,
without any human pointing at it.

We implement change-point detection from scratch (binary segmentation)
so the logic is fully transparent - no black-box library.

Core idea:
  For a candidate split point t, model the series as two segments:
  [0:t] and [t:end], each represented by its own mean.
  The "cost" of a segmentation is the sum of squared errors (SSE)
  of each point from its segment's mean.
  The best split point t* is the one that minimizes total SSE
  (i.e. explains the data better than treating it as one flat segment).
"""

import numpy as np
import pandas as pd


def _sse(values: np.ndarray) -> float:
    """Sum of squared errors from the segment's own mean."""
    if len(values) == 0:
        return 0.0
    return float(np.sum((values - values.mean()) ** 2))


def detect_change_point(series: pd.Series, min_segment_size: int = 10) -> dict:
    """
    Finds the single most significant change point in a 1D series.
    """
    values = series.values
    n = len(values)

    baseline_sse = _sse(values)  # cost of NOT splitting at all

    best_split = None
    best_sse = np.inf

    for t in range(min_segment_size, n - min_segment_size):
        left, right = values[:t], values[t:]
        total_sse = _sse(left) + _sse(right)
        if total_sse < best_sse:
            best_sse = total_sse
            best_split = t

    before_mean = values[:best_split].mean()
    after_mean = values[best_split:].mean()
    pct_change = (after_mean - before_mean) / before_mean * 100

    confidence = max(0.0, (baseline_sse - best_sse) / baseline_sse) if baseline_sse > 0 else 0.0

    return {
        "change_point_index": int(best_split),
        "change_point_date": series.index[best_split],
        "before_mean": round(float(before_mean), 2),
        "after_mean": round(float(after_mean), 2),
        "pct_change": round(float(pct_change), 2),
        "confidence": round(float(confidence), 3),
    }


def detect_anomaly(df: pd.DataFrame, date_col: str, target_col: str) -> dict:
    """
    High-level entry point: takes a dataframe + target metric name,
    returns a structured anomaly report describing WHAT changed and WHEN.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)

    result = detect_change_point(df[target_col])

    direction = "dropped" if result["pct_change"] < 0 else "increased"

    summary = (
        f"'{target_col}' {direction} by {abs(result['pct_change'])}% "
        f"(from {result['before_mean']} to {result['after_mean']}) "
        f"around {result['change_point_date'].date()}, "
        f"detection confidence: {result['confidence']*100:.1f}%"
    )

    return {
        "metric": target_col,
        "direction": direction,
        **result,
        "summary": summary,
    }


if __name__ == "__main__":
    df = pd.read_csv("data/sales_data.csv")
    anomaly = detect_anomaly(df, date_col="date", target_col="sales")
    print("ANOMALY DETECTED:")
    for k, v in anomaly.items():
        print(f"  {k}: {v}")