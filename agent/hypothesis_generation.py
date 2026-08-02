"""
hypothesis_generation.py
-------------------------
The agent's "reasoning" step: given an anomaly summary and the columns
available in the dataset, ask the LLM to propose CANDIDATE causes to
investigate. These are just hypotheses at this stage - nothing is
confirmed yet. Phase 4 (causal testing) will check which ones actually hold up.
"""

import json
import ollama

MODEL = "llama3.2:3b"

HYPOTHESIS_PROMPT = """You are a data analyst investigating a business metric anomaly.

ANOMALY DETECTED:
{anomaly_summary}

AVAILABLE DATA COLUMNS (other variables you could investigate as possible causes):
{columns}

Your task: propose 3 to 5 candidate hypotheses for what might have caused this anomaly.
For each hypothesis, ONLY use the column names given above as the proposed cause.

Respond ONLY with valid JSON, no other text, in exactly this format:
{{
  "hypotheses": [
    {{
      "cause_column": "column_name_here",
      "reasoning": "one sentence explaining why this could plausibly be the cause"
    }}
  ]
}}
"""


def generate_hypotheses(anomaly_summary: str, available_columns: list[str]) -> dict:
    prompt = HYPOTHESIS_PROMPT.format(
        anomaly_summary=anomaly_summary,
        columns=", ".join(available_columns),
    )

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",  # ask Ollama to constrain output to valid JSON
    )

    raw = response["message"]["content"]

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        print("WARNING: model did not return valid JSON. Raw output:")
        print(raw)
        parsed = {"hypotheses": []}

    return parsed


if __name__ == "__main__":
    # Quick manual test using our Step 7 anomaly output
    test_summary = (
        "'sales' dropped by 16.06% (from 637.68 to 535.3) around 2026-05-06, "
        "detection confidence: 82.9%"
    )
    test_columns = ["marketing_spend", "foot_traffic", "competitor_price", "our_price"]

    result = generate_hypotheses(test_summary, test_columns)
    print(json.dumps(result, indent=2))