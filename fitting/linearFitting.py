import pandas as pd
import numpy as np
import glob
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Output setup
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "slippage_linear_summary.csv")

# Initialize summary DataFrame
summary_df = pd.DataFrame(columns=["ticker", "slope", "intercept", "r2"])

# Get list of .parquet files excluding summary files
parquet_files = [
    f for f in glob.glob(os.path.join(".", "*.parquet"))
    if "summary" not in f
]

for file_path in parquet_files:
    df = pd.read_parquet(file_path)

    # Extract slippage columns
    slippage_cols = [col for col in df.columns if col.startswith("Slippage_")]
    if not slippage_cols:
        print(f"⚠ Skipping {file_path}: No Slippage columns found")
        continue

    try:
        order_sizes = [int(col.split("_")[1]) for col in slippage_cols]
        slippage_means = df[slippage_cols].mean().values

        x = np.array(order_sizes).reshape(-1, 1)
        y = slippage_means

        model = LinearRegression().fit(x, y)
        y_pred = model.predict(x)
        r2 = r2_score(y, y_pred)

        ticker = df["Ticker"].iloc[0] if "Ticker" in df.columns else os.path.basename(file_path)

        summary_df = pd.concat([summary_df, pd.DataFrame([{
            "ticker": ticker,
            "slope": model.coef_[0],
            "intercept": model.intercept_,
            "r2": r2
        }])])

        print(f"✔ Fitted linear model for {ticker} | R² = {r2:.4f}")
    except Exception as e:
        print(f"❌ Failed linear fit for {file_path}: {e}")

# Save summary as CSV
summary_df.to_csv(SUMMARY_FILE, index=False)
print(f"\n📄 Linear model summary saved to {SUMMARY_FILE}")

# Compute and display average R²
if not summary_df.empty:
    avg_r2 = summary_df["r2"].mean()
    print(f"📊 Average R² across all tickers: {avg_r2:.4f}")
else:
    print("⚠ No successful fits to compute average R².")
