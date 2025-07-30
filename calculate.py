import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

# === CONFIGURATION ===
order_sizes = np.arange(100, 2001, 100)

ask_px_cols = [f"ask_px_0{i}" for i in range(10)]
ask_sz_cols = [f"ask_sz_0{i}" for i in range(10)]
bid_px_cols = [f"bid_px_0{i}" for i in range(10)]
bid_sz_cols = [f"bid_sz_0{i}" for i in range(10)]

# Include necessary columns to compute mid-price
required_cols = ask_px_cols + ask_sz_cols + bid_px_cols + bid_sz_cols + ["ask_px_00", "bid_px_00"]

# === FUNCTION: Compute slippage for a single row ===
def compute_slippage_row(row_dict, side="buy"):
    if side == "buy":
        px_cols = ask_px_cols
        sz_cols = ask_sz_cols
        best_bid = float(row_dict.get("bid_px_00", np.nan))
        best_ask = float(row_dict.get("ask_px_00", np.nan))
    elif side == "sell":
        px_cols = bid_px_cols
        sz_cols = bid_sz_cols
        best_bid = float(row_dict.get("bid_px_00", np.nan))
        best_ask = float(row_dict.get("ask_px_00", np.nan))
    else:
        raise ValueError("Invalid side. Use 'buy' or 'sell'.")

    if np.isnan(best_bid) or np.isnan(best_ask):
        return {f"Slippage_{size}": np.nan for size in order_sizes}

    mid_price = (best_bid + best_ask) / 2
    slippage_data = {}

    for size in order_sizes:
        shares_needed = size
        total_cost = 0

        for px_col, sz_col in zip(px_cols, sz_cols):
            try:
                price = float(row_dict[px_col])
                available = float(row_dict[sz_col])
            except (KeyError, ValueError):
                continue

            if shares_needed <= 0:
                break

            filled = min(shares_needed, available)
            total_cost += filled * price
            shares_needed -= filled

        if shares_needed > 0:
            slippage = np.nan
        else:
            avg_price = total_cost / size
            slippage = avg_price - mid_price if side == "buy" else mid_price - avg_price

        slippage_data[f"Slippage_{size}"] = slippage

    return slippage_data

# === FUNCTION: Process a ticker folder ===
def process_ticker_folder(ticker_folder, ticker_name, row_id_start):
    results = []
    all_files = sorted([f for f in os.listdir(ticker_folder) if f.endswith('.csv')])
    row_id = row_id_start

    for file in tqdm(all_files, desc=f"Processing {ticker_name}"):
        filepath = os.path.join(ticker_folder, file)
        df = pd.read_csv(filepath, usecols=required_cols)

        for row in df.itertuples(index=False):
            row_dict = row._asdict()
            slippage_row = compute_slippage_row(row_dict, side="buy")
            slippage_row.update({
                "Ticker": ticker_name,
                "File_Name": file,
                "Row_ID": row_id,
                "Date": file.split('_')[1] if '_' in file else "Unknown"
            })
            results.append(slippage_row)
            row_id += 1

    return results, row_id

# === FUNCTION: Process all tickers and stream to Parquet ===
def process_all_tickers_streaming(base_dir):
    row_id = 0
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            print(f"🟢 Processing {folder}")
            ticker_data, row_id = process_ticker_folder(folder_path, folder, row_id)

            df_ticker = pd.DataFrame.from_records(ticker_data)
            df_ticker.reset_index(drop=True, inplace=True)

            table = pa.Table.from_pandas(df_ticker)
            pq.write_table(table, f"slippage_{folder}.parquet", compression=None)
            print(f"✅ Saved {folder} to slippage_{folder}.parquet")

# === MAIN ===
if __name__ == "__main__":
    base_dir = "."
    process_all_tickers_streaming(base_dir)
