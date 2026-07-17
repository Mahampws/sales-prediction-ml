"""
Generates a synthetic retail sales dataset for demo/training purposes.
Replace this with a client's real historical sales data (CSV) for actual use.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000

ad_spend = np.random.uniform(500, 10000, n)
store_traffic = np.random.uniform(50, 2000, n)
discount_pct = np.random.uniform(0, 40, n)
is_holiday_season = np.random.choice([0, 1], n, p=[0.8, 0.2])
competitor_price_diff = np.random.uniform(-20, 20, n)

# Simulated relationship with noise
sales = (
    50 * ad_spend / 100
    + 15 * store_traffic
    + 200 * discount_pct
    + 5000 * is_holiday_season
    - 80 * competitor_price_diff
    + np.random.normal(0, 3000, n)
)
sales = np.clip(sales, 0, None)

df = pd.DataFrame(
    {
        "ad_spend": ad_spend.round(2),
        "store_traffic": store_traffic.round(0),
        "discount_pct": discount_pct.round(1),
        "is_holiday_season": is_holiday_season,
        "competitor_price_diff": competitor_price_diff.round(2),
        "sales": sales.round(2),
    }
)

df.to_csv("sales_data.csv", index=False)
print(f"Generated {len(df)} rows -> sales_data.csv")
