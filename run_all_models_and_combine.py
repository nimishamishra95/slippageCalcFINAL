import pandas as pd
import subprocess
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
OUTPUT_DIR = "output"
FINAL_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "slippage_all_models_summary.csv")

# Run model scripts
scripts = [
    "fitting/parametricFitting.py",
    "fitting/linearFitting.py",
    "fitting/quadraticFitting.py",
    "fitting/nonparametricFitting.py"
]

for script in scripts:
    print(f"🚀 Running {script}...")
    subprocess.run(["python", script], check=True)

# Load results
def load_with_model_type(path, model):
    df = pd.read_csv(path)
    df["model"] = model
    return df

try:
    df_parametric = load_with_model_type(os.path.join(OUTPUT_DIR, "slippage_parametric_summary.csv"), "parametric")
    df_linear = load_with_model_type(os.path.join(OUTPUT_DIR, "slippage_linear_summary.csv"), "linear")
    df_quadratic = load_with_model_type(os.path.join(OUTPUT_DIR, "slippage_quadratic_summary.csv"), "quadratic")

    df_nonparam = pd.read_csv(os.path.join(OUTPUT_DIR, "slippage_nonparametric_summary.csv"))

    # Map 'method' from nonparam file to 'model' column
    df_nonparam["model"] = df_nonparam["method"].str.lower()
    df_nonparam = df_nonparam[["ticker", "model", "r2"]]  # Keep only consistent columns

    # Align parametric models to same column set
    df_parametric = df_parametric[["ticker", "model", "r2"]]
    df_linear = df_linear[["ticker", "model", "r2"]]
    df_quadratic = df_quadratic[["ticker", "model", "r2"]]

    # Combine all
    combined_df = pd.concat([df_parametric, df_linear, df_quadratic, df_nonparam], ignore_index=True)

    combined_df.to_csv(FINAL_OUTPUT_FILE, index=False)
    print(f"\n✅ Combined model results saved to {FINAL_OUTPUT_FILE}")

    # Error analysis
    r2_summary = combined_df.groupby("model")["r2"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    r2_summary.to_csv(os.path.join(OUTPUT_DIR, "r2_summary_by_model.csv"), index=False)
    print("\n📊 R² Summary by Model:\n", r2_summary)

    # R² boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=combined_df, x="model", y="r2", palette="Set2")
    plt.title("R² Distribution by Model Type")
    plt.xlabel("Model Type")
    plt.ylabel("R² Score")
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "r2_boxplot_by_model.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"📸 R² boxplot saved to {plot_path}")

except Exception as e:
    print(f"❌ Error during model aggregation or visualization: {e}")
