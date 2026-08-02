"""
causal_testing.py
-------------------
The agent's "verification" step: takes candidate hypotheses from Phase 3
and statistically tests each one using Granger causality.

IMPORTANT LESSON LEARNED: raw Granger causality on non-stationary data
(data with a trend/level-shift, like ours) gives FALSE POSITIVES - every
variable that shifts around the same time looks "causal," even unrelated
ones. This is the same trap as naive correlation, just disguised.

FIX: check stationarity with the Augmented Dickey-Fuller (ADF) test, and
if a series is non-stationary, DIFFERENCE it (convert to day-over-day
change) before running Granger causality. This removes shared trends and
leaves only genuine short-term predictive relationships.
"""

import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

MAX_LAG = 5
SIGNIFICANCE = 0.05


def is_stationary(series: pd.Series) -> bool:
    """Augmented Dickey-Fuller test: low p-value (<0.05) means stationary."""
    result = adfuller(series.dropna())
    p_value = result[1]
    return p_value < SIGNIFICANCE


def make_stationary(series: pd.Series) -> pd.Series:
    """Differences a series (day-over-day change) to remove trend."""
    return series.diff().dropna()

def _classify_evidence(p_value: float) -> str:
    """
    Converts a raw p-value into a human-meaningful evidence tier, instead
    of a blunt yes/no cutoff. This matters because p=0.0001 and p=0.048
    would both just say "SUPPORTED" under a binary threshold, even though
    the strength of evidence is wildly different.
    """
    if p_value < 0.001:
        return "STRONG EVIDENCE"
    elif p_value < 0.01:
        return "MODERATE EVIDENCE"
    elif p_value < 0.05:
        return "WEAK EVIDENCE"
    else:
        return "NOT SUPPORTED"
    
def test_hypothesis(df: pd.DataFrame, cause_col: str, target_col: str = "sales") -> dict:
    target = df[target_col]
    cause = df[cause_col]

    target_stationary = is_stationary(target)
    cause_stationary = is_stationary(cause)

    # difference whichever series need it, so both are stationary before testing
    target_clean = target if target_stationary else make_stationary(target)
    cause_clean = cause if cause_stationary else make_stationary(cause)

    data = pd.concat([target_clean, cause_clean], axis=1).dropna()
    data.columns = [target_col, cause_col]

    try:
        results = grangercausalitytests(data, maxlag=MAX_LAG, verbose=False)
    except Exception as e:
        return {"cause_column": cause_col, "error": str(e), "verdict": "TEST_FAILED"}

    best_p = min(results[lag][0]["ssr_ftest"][1] for lag in results)
    best_lag = min(results, key=lambda lag: results[lag][0]["ssr_ftest"][1])

    correlation = df[target_col].corr(df[cause_col])

    return {
        "cause_column": cause_col,
        "p_value": round(best_p, 4),
        "best_lag_days": best_lag,
        "correlation": round(correlation, 3),
        "was_differenced": not (target_stationary and cause_stationary),
        "verdict": _classify_evidence(best_p),
    }


def test_all_hypotheses(df: pd.DataFrame, hypotheses: list[dict], target_col: str = "sales") -> list[dict]:
    tested = []
    for h in hypotheses:
        cause = h["cause_column"]
        result = test_hypothesis(df, cause, target_col)
        result["reasoning"] = h.get("reasoning", "")
        tested.append(result)
    tested.sort(key=lambda r: r.get("p_value", 1.0))
    return tested


if __name__ == "__main__":
    df = pd.read_csv("data/sales_data.csv")

    test_hypotheses = [
        {"cause_column": "marketing_spend", "reasoning": "Marketing spend drop"},
        {"cause_column": "foot_traffic", "reasoning": "Foot traffic decline"},
        {"cause_column": "competitor_price", "reasoning": "Competitor price cut"},
        {"cause_column": "our_price", "reasoning": "Our price too high"},
    ]

    results = test_all_hypotheses(df, test_hypotheses)

    print("CAUSAL TEST RESULTS (after correcting for non-stationarity):\n")
    for r in results:
        print(f"  {r['cause_column']:<20} p-value: {r.get('p_value', 'N/A'):<8} "
              f"correlation: {r.get('correlation', 'N/A'):<8} "
              f"differenced: {r.get('was_differenced', 'N/A'):<6} verdict: {r['verdict']}")