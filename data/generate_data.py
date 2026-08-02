"""
generate_data.py
-----------------
Generates a synthetic business dataset with a KNOWN root cause, so we can
verify our causal agent actually finds the truth (not just something plausible).

The story:
- We're a retailer tracking daily sales for Region X over 180 days.
- Around day 120, a competitor drops their price -> this is the TRUE cause
  of our sales decline (people switch to the competitor).
- Around the same time, there's also a seasonal dip (fewer shoppers in general)
  and a marketing spend cut -> these are CONFOUNDERS / red herrings that a
  naive analyst (or naive LLM) might blame instead.

Columns:
- date
- sales                (target metric - what "broke")
- marketing_spend      (red herring #1 - correlated but not causal)
- foot_traffic         (seasonal factor - a genuine confounder)
- competitor_price     (TRUE cause - inversely drives our sales)
- our_price            (control variable, stays constant)
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_DAYS = 180
dates = pd.date_range("2026-01-01", periods=N_DAYS, freq="D")

# 1. Seasonal foot traffic (genuine confounder): dips slightly in a "low season"
#    from day 110 to 160, plus weekly seasonality (weekends busier)
day_of_week = dates.dayofweek
weekend_boost = np.where(day_of_week >= 5, 15, 0)
season_dip = np.where((np.arange(N_DAYS) >= 110) & (np.arange(N_DAYS) <= 160), -10, 0)
foot_traffic = 100 + weekend_boost + season_dip + np.random.normal(0, 5, N_DAYS)

# 2. Marketing spend (red herring): company coincidentally cut marketing
#    budget around day 120 for unrelated budget-cycle reasons. It has only
#    a WEAK true effect on sales, but the timing makes it look suspicious.
marketing_spend = np.where(np.arange(N_DAYS) >= 120, 800, 1000) + np.random.normal(0, 30, N_DAYS)

# 3. Competitor price (TRUE cause): stable, then drops sharply around day 125
competitor_price = np.where(np.arange(N_DAYS) >= 125, 15, 20) + np.random.normal(0, 0.5, N_DAYS)

# 4. Our price: stays flat (we did NOT react in time - that's the business problem)
our_price = 22 + np.random.normal(0, 0.3, N_DAYS)

# 5. Sales generation (ground truth causal model):
#    sales depends on foot_traffic (weakly), marketing (weakly),
#    and STRONGLY, inversely, on the price GAP between us and competitor.
#    IMPORTANT: customers don't react to a competitor's price change instantly -
#    they take a few days to notice and switch. We model this with a 3-day lag,
#    which is also what makes Granger causality (a LAGGED-relationship test)
#    able to correctly detect this as the true cause.
REACTION_LAG_DAYS = 3
competitor_price_lagged = pd.Series(competitor_price).shift(REACTION_LAG_DAYS).bfill().values
price_gap = our_price - competitor_price_lagged  # bigger gap = we're more expensive = fewer sales

base_sales = 500
sales = (
    base_sales
    + 1.2 * foot_traffic          # weak positive driver
    + 0.05 * marketing_spend      # very weak driver (red herring effect is tiny)
    - 18 * price_gap              # STRONG driver - this is the true mechanism (now lagged 3 days)
    + np.random.normal(0, 15, N_DAYS)  # noise
)

df = pd.DataFrame({
    "date": dates,
    "sales": sales.round(1),
    "marketing_spend": marketing_spend.round(1),
    "foot_traffic": foot_traffic.round(1),
    "competitor_price": competitor_price.round(2),
    "our_price": our_price.round(2),
})

df.to_csv("data/sales_data.csv", index=False)

print("Dataset generated: data/sales_data.csv")
print(df.head())
print("\nGROUND TRUTH (for your own verification later, don't show the agent this):")
print("- TRUE root cause: competitor_price drop around day 125 widened our price gap")
print("- RED HERRING #1: marketing_spend also dropped around day 120 (coincidental timing)")
print("- CONFOUNDER: foot_traffic dipped (day 110-160) due to season, affecting sales independently")