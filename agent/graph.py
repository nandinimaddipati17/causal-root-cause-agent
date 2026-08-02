"""
graph.py
---------
Wires anomaly_detection, hypothesis_generation, and causal_testing into
a single orchestrated LangGraph agent.

State flows through 3 nodes in sequence:
  detect_anomaly -> generate_hypotheses -> test_hypotheses

Each node reads what it needs from the shared state and writes its
results back into it, so the next node can use them.
"""

from typing import TypedDict
import pandas as pd
from langgraph.graph import StateGraph, END

from anomaly_detection import detect_anomaly
from hypothesis_generation import generate_hypotheses
from causal_testing import test_all_hypotheses


class AgentState(TypedDict):
    data_path: str
    date_col: str
    target_col: str
    anomaly: dict
    hypotheses: list
    causal_results: list


def node_detect_anomaly(state: AgentState) -> dict:
    print(">> [Node 1] Detecting anomaly...")
    df = pd.read_csv(state["data_path"])
    anomaly = detect_anomaly(df, state["date_col"], state["target_col"])
    print(f"   {anomaly['summary']}")
    return {"anomaly": anomaly}


def node_generate_hypotheses(state: AgentState) -> dict:
    print(">> [Node 2] Generating hypotheses (LLM reasoning)...")
    df = pd.read_csv(state["data_path"])
    available_columns = [c for c in df.columns if c not in (state["date_col"], state["target_col"])]

    result = generate_hypotheses(state["anomaly"]["summary"], available_columns)
    hypotheses = result.get("hypotheses", [])
    print(f"   Generated {len(hypotheses)} candidate hypotheses")
    return {"hypotheses": hypotheses}


def node_test_hypotheses(state: AgentState) -> dict:
    print(">> [Node 3] Testing hypotheses (Granger causality)...")
    df = pd.read_csv(state["data_path"])
    results = test_all_hypotheses(df, state["hypotheses"], state["target_col"])
    print(f"   Tested {len(results)} hypotheses")
    return {"causal_results": results}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("detect_anomaly", node_detect_anomaly)
    graph.add_node("generate_hypotheses", node_generate_hypotheses)
    graph.add_node("test_hypotheses", node_test_hypotheses)

    graph.set_entry_point("detect_anomaly")
    graph.add_edge("detect_anomaly", "generate_hypotheses")
    graph.add_edge("generate_hypotheses", "test_hypotheses")
    graph.add_edge("test_hypotheses", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    initial_state = {
        "data_path": "data/sales_data.csv",
        "date_col": "date",
        "target_col": "sales",
    }

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 60)
    print("FINAL RESULTS (ranked by strength of causal evidence):")
    print("=" * 60)
    for r in final_state["causal_results"]:
        print(f"  {r['cause_column']:<20} verdict: {r['verdict']}")