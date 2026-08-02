"""
app.py
-------
Streamlit UI for the Causal Root-Cause Analysis Agent.
Lets a user pick a metric, run the full agent pipeline, and see:
  - the detected anomaly
  - ranked causal hypotheses with evidence strength
  - the final human-readable report

Run with: streamlit run ui/app.py
"""

import sys
import os

# make the agent/ folder importable from here
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "agent"))

import streamlit as st
import pandas as pd
from graph import build_graph
from report_generation import generate_report

st.set_page_config(page_title="Causal Root-Cause Agent", page_icon="🔎", layout="centered")

st.title("🔎 Causal Root-Cause Analysis Agent")
st.caption("Give it a business metric. It investigates the anomaly, tests causal hypotheses, and reports the real root cause — not just what correlates.")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sales_data.csv")

df_preview = pd.read_csv(DATA_PATH)
columns = [c for c in df_preview.columns if c != "date"]

with st.sidebar:
    st.header("Configuration")
    target_col = st.selectbox("Metric to investigate", columns, index=columns.index("sales") if "sales" in columns else 0)
    st.divider()
    st.caption("Dataset preview")
    st.dataframe(df_preview.head(5), use_container_width=True)

run_button = st.button("🚀 Run Root Cause Analysis", type="primary", use_container_width=True)

if run_button:
    with st.spinner("Agent investigating... (running local LLM, may take 20-40 sec)"):
        app = build_graph()
        initial_state = {
            "data_path": DATA_PATH,
            "date_col": "date",
            "target_col": target_col,
        }
        final_state = app.invoke(initial_state)

    st.success("Investigation complete")

    # --- Anomaly section ---
    st.subheader("📉 Detected Anomaly")
    anomaly = final_state["anomaly"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Change", f"{anomaly['pct_change']}%")
    col2.metric("Detection Confidence", f"{anomaly['confidence']*100:.0f}%")
    col3.metric("Change Point", str(anomaly["change_point_date"])[:10])
    st.write(anomaly["summary"])

    # --- Causal results section ---
    st.subheader("🧪 Causal Testing Results")
    results = final_state["causal_results"]

    def verdict_color(v):
        return {
            "STRONG EVIDENCE": "🟢",
            "MODERATE EVIDENCE": "🟡",
            "WEAK EVIDENCE": "🟠",
            "NOT SUPPORTED": "🔴",
        }.get(v, "⚪")

    results_df = pd.DataFrame(results)
    results_df["evidence"] = results_df["verdict"].apply(verdict_color) + " " + results_df["verdict"]
    display_cols = ["cause_column", "p_value", "correlation", "evidence"]
    display_cols = [c for c in display_cols if c in results_df.columns]
    st.dataframe(results_df[display_cols], use_container_width=True, hide_index=True)

    # --- Report section ---
    st.subheader("📝 Root Cause Report")
    with st.spinner("Writing report..."):
        report = generate_report(anomaly["summary"], results)
    st.markdown(report)