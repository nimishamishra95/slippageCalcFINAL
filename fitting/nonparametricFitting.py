import pandas as pd
import numpy as np
import glob
import os
from sklearn.metrics import r2_score
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import UnivariateSpline

# Output setup
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "slippage_nonparametric_summary.csv")

# Initialize summary DataFrame
summary_df = pd.DataFrame(columns=["ticker", "method", "smoothing_param", "spline_s", "r2"])

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
        order_sizes = np.array([int(col.split("_")[1]) for col in slippage_cols])
        slippage_means = df[slippage_cols].mean().values

        x = order_sizes
        y = slippage_means
        ticker = df["Ticker"].iloc[0] if "Ticker" in df.columns else os.path.basename(file_path)

        ## LOWESS ##
        frac = 0.5  # smoothing parameter (you can tune)
        lowess_fit = lowess(y, x, frac=frac, return_sorted=False)
        r2_lowess = r2_score(y, lowess_fit)
        summary_df = pd.concat([summary_df, pd.DataFrame([{
            "ticker": ticker,
            "method": "LOWESS",
            "smoothing_param": frac,
            "spline_s": np.nan,
            "r2": r2_lowess
        }])])
        print(f"✔ LOWESS fit for {ticker} | R² = {r2_lowess:.4f}")

        ## Univariate Spline ##
        s_param = 1  # smoothing factor (tunable)
        spline = UnivariateSpline(x, y, s=s_param)
        y_spline = spline(x)
        r2_spline = r2_score(y, y_spline)
        summary_df = pd.concat([summary_df, pd.DataFrame([{
            "ticker": ticker,
            "method": "Spline",
            "smoothing_param": np.nan,
            "spline_s": s_param,
            "r2": r2_spline
        }])])
        print(f"✔ Spline fit for {ticker} | R² = {r2_spline:.4f}")

    except Exception as e:
        print(f"❌ Failed non-parametric fit for {file_path}: {e}")

# Save summary
summary_df.to_csv(SUMMARY_FILE, index=False)
print(f"\n📄 Non-parametric model summary saved to {SUMMARY_FILE}")

# Print overall R² stats
if not summary_df.empty:
    avg_r2 = summary_df.groupby("method")["r2"].mean()
    print(f"\n📊 Average R² by method:\n{avg_r2}")
else:
    print("⚠ No successful fits to compute average R².")
