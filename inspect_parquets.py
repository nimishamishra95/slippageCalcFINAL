import pandas as pd
import glob
import os

def print_each_parquet(parquet_dir="."):
    # Find all .parquet files in the directory
    parquet_files = glob.glob(os.path.join(parquet_dir, "slippage_*.parquet"))

    if not parquet_files:
        print("❌ No parquet files found.")
        return

    for file in parquet_files:
        print(f"\n🗂️ File: {file}")
        try:
            df = pd.read_parquet(file)
            print(df.head(10))  # Print only first 5 rows to keep it simple
        except Exception as e:
            print(f"⚠️ Error reading {file}: {e}")

if __name__ == "__main__":
    print_each_parquet()
