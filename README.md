**Add ticker folders into repo before running (file size too large to push to git even after using git lfs)**
 FILE STRUCTURE SHOULD BE:
   slippageCalcFinal
     **- CRWV (folder with all CRWV csv files)
     - FROG (folder with all FROG csv files)
     - SOUN (folder with all SOUN csv files)**
     - output (will populate)
     - fitting (scripts - leave untouched)
     - calculate.py
     - run_all_models_and_combine.py
     - inspect_parquets.py
     - (some .parquet files that will populate later)

RUN:
  1. calculate.py
  2. run_all_models_and_combine.py

VIEW output folder for analysis
