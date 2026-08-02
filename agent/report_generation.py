"""
report_generation.py
----------------------
The agent's "communication" step: takes the anomaly + tested hypotheses
and asks the LLM to write a clear, business-readable root cause report -
citing the actual evidence, not just vibes.
"""

import ollama

MODEL = "llama3.2:3b"

REPORT_PROMPT = """You are a data analyst writing a root cause analysis report for a business stakeholder.

ANOMALY:
{anomaly_summary}

STATISTICALLY TESTED HYPOTHESES (ranked by strength of evidence, from Granger causality testing):
{results_text}

Write a concise root cause report (max 200 words) for a non-technical business audience. Structure:
1. What happened (the anomaly, in plain language)
2. The most likely root cause (based on the STRONGEST evidence only)
3. Briefly mention what was investigated but ruled out or found weak, and why
4. One concrete recommended action

Be direct and confident where evidence is strong ("STRONG EVIDENCE"), and appropriately cautious
where evidence is weak. Do NOT present weak or unsupported hypotheses as if they were confirmed causes.
"""


def format_results_for_prompt(results: list[dict]) -> str:
    lines = []
    for r in results:
        lines.append(
            f"- {r['cause_column']}: {r['verdict']} (p-value: {r.get('p_value', 'N/A')}, "
            f"correlation: {r.get('correlation', 'N/A')})"
        )
    return "\n".join(lines)


def generate_report(anomaly_summary: str, causal_results: list[dict]) -> str:
    results_text = format_results_for_prompt(causal_results)
    prompt = REPORT_PROMPT.format(anomaly_summary=anomaly_summary, results_text=results_text)

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    return response["message"]["content"]


if __name__ == "__main__":
    test_summary = "'sales' dropped by 16.06% (from 637.55 to 535.17) around 2026-05-09, detection confidence: 82.9%"
    test_results = [
        {"cause_column": "competitor_price", "verdict": "STRONG EVIDENCE", "p_value": 0.0, "correlation": 0.862},
        {"cause_column": "marketing_spend", "verdict": "WEAK EVIDENCE", "p_value": 0.0482, "correlation": 0.821},
        {"cause_column": "foot_traffic", "verdict": "NOT SUPPORTED", "p_value": 0.1179, "correlation": 0.381},
    ]

    report = generate_report(test_summary, test_results)
    print("=" * 60)
    print("ROOT CAUSE REPORT")
    print("=" * 60)
    print(report)