import pandas as pd
import numpy as np
import glob
import os
from sklearn.metrics import r2_score

# Quadratic model: slippage = a*x^2 + b*x + c
def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

# Output setup
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "slippage_quadratic_summary.csv")

# Initialize summary DataFrame
summary_df = pd.DataFrame(columns=["ticker", "a", "b", "c", "r2"])

# Get list of .parquet files excluding summaries
parquet_files = [
    f for f in glob.glob(os.path.join(".", "*.parquet"))
    if "summary" not in f
]

for file_path in parquet_files:
    df = pd.read_parquet(file_path)

    slippage_cols = [col for col in df.columns if col.startswith("Slippage_")]
    if not slippage_cols:
        print(f"⚠ Skipping {file_path}: No Slippage columns found")
        continue

    try:
        order_sizes = [int(col.split("_")[1]) for col in slippage_cols]
        slippage_means = df[slippage_cols].mean().values

        x = np.array(order_sizes)
        y = slippage_means

        coeffs = np.polyfit(x, y, deg=2)  # Fit quadratic
        y_pred = np.polyval(coeffs, x)
        r2 = r2_score(y, y_pred)

        ticker = df["Ticker"].iloc[0] if "Ticker" in df.columns else os.path.basename(file_path)

        summary_df = pd.concat([summary_df, pd.DataFrame([{
            "ticker": ticker,
            "a": coeffs[0],
            "b": coeffs[1],
            "c": coeffs[2],
            "r2": r2
        }])])

        print(f"✔ Fitted quadratic model for {ticker} | R² = {r2:.4f}")
    except Exception as e:
        print(f"❌ Failed quadratic fit for {file_path}: {e}")

# Save summary as CSV
summary_df.to_csv(SUMMARY_FILE, index=False)
print(f"\n📄 Quadratic model summary saved to {SUMMARY_FILE}")

# Compute and display average R²
if not summary_df.empty:
    avg_r2 = summary_df["r2"].mean()
    print(f"📊 Average R² across all tickers: {avg_r2:.4f}")
else:
    print("⚠ No successful fits to compute average R².")
