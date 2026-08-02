# 🔎 Causal Root-Cause Analysis Agent

An AI agent that investigates business metric anomalies the way a rigorous analyst would — not by chasing the most dramatic-looking correlation, but by statistically testing which candidate causes actually hold up.

Most "AI analytics" tools stop at *"here's what correlates with your metric."* This project goes further: it detects anomalies, generates hypotheses using a local LLM, tests each one with **Granger causality**, and rejects the misleading ones — before writing a business-readable root cause report.

## 🧠 The Core Problem

Dashboards tell you *what* happened. They rarely tell you *why* — and worse, the most visually correlated variable is often not the true cause. This project was built and validated against a synthetic dataset with a **known, hidden root cause** plus deliberate red herrings, specifically to prove the agent finds the truth rather than the most convenient story.

**The validation story:**
- A retailer's sales dropped 16% around a specific date
- Naive correlation ranked `marketing_spend` (0.85) almost as suspicious as the actual cause, `competitor_price` (0.93)
- A first pass with Granger causality *also* got fooled — non-stationary data caused every variable to appear "significant," a well-known pitfall in time-series causal inference
- After correcting for stationarity and adding evidence-strength tiers (rather than a blunt pass/fail threshold), the agent correctly separated:
  - `competitor_price` → **STRONG EVIDENCE** (p ≈ 0.0) — the true cause
  - `marketing_spend` → **WEAK EVIDENCE** (p = 0.048) — the red herring, correctly demoted
  - `foot_traffic` → **NOT SUPPORTED** (p = 0.12) — the confounder, correctly rejected

## 🏗️ Architecture

Orchestrated as a multi-node **LangGraph** agent, with a **Streamlit** UI for interactive investigation.

## 🛠️ Tech Stack

- **Orchestration:** LangGraph
- **LLM:** Llama 3.2 (3B), served locally and free via Ollama
- **Causal inference:** Granger causality (statsmodels), stationarity testing (ADF)
- **Data:** Pandas, NumPy
- **UI:** Streamlit

## 🚀 Running it locally

```bash
# 1. Clone and set up environment
git clone https://github.com/nandinimaddipati17/causal-root-cause-agent.git
cd causal-root-cause-agent
python -m venv venv
venv\Scripts\activate.bat   # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install Ollama (https://ollama.com) and pull the model
ollama pull llama3.2:3b

# 4. Generate the synthetic dataset
python data/generate_data.py

# 5. Run the agent pipeline directly
python agent/graph.py

# 6. Or launch the interactive UI
streamlit run ui/app.py
```

## 📂 Project Structure

## 💡 Key Learnings

- Naive correlation is actively misleading when multiple variables shift around the same time — a genuinely common real-world scenario (budget cycles, seasonality, and the actual cause often overlap)
- Granger causality assumes stationary data; skipping this check produces false positives that *look* rigorous but aren't
- Granger causality tests **lagged** relationships — a same-day causal effect requires the underlying mechanism to reflect realistic response delay to be detectable
- A binary "significant / not significant" threshold hides useful information; evidence-strength tiers give a more honest, actionable picture