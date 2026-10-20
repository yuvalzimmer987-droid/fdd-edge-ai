import os
import shutil
import numpy as np
import pandas as pd
from detector import Detector
from preprocessing import DATA_DIR, load_plant_file, day_split, steady_state_mask

# Small data pack for the online dashboard (Streamlit Community Cloud).
#
# The full LBNL files are too large for GitHub, so this keeps only what the
# dashboard shows: the test days, steady chiller operation, and the columns
# the two detectors use - for all 24 files. dashboard.py reads this pack
# when the full data folder is not there.
#
# Run: python src/make_demo_data.py      (after subsystems.py --hybrid --sweep
#                                          and export_model.py)

OUT = "demo_data"
os.makedirs(OUT, exist_ok=True)

detector = Detector.from_saved()
columns = detector.load_columns()
catalog = pd.read_csv("data/file_catalog.csv", dtype={"severity": str})

total = 0
for filename in catalog["filename"]:
    df = load_plant_file(f"{DATA_DIR}/{filename}", columns)    # OA columns already swapped back
    keep, _, _, _ = steady_state_mask(df)
    keep = keep & (day_split(len(df)) == "test")
    part = df.iloc[np.flatnonzero(keep)].astype(np.float32)      # index = row number in the file
    path = f"{OUT}/{filename.replace('.csv', '.parquet')}"
    part.to_parquet(path, compression="zstd")
    size = os.path.getsize(path) / 1e6
    total += size
    print(f"{filename:62s} {len(part):6d} rows  {size:5.2f} MB")

# Results shown in the dashboard's tiles
catalog.to_csv(f"{OUT}/file_catalog.csv", index=False)
for name in ["hybrid_tradeoff.csv", "edge_results.csv"]:
    if os.path.exists(f"data/{name}"):
        shutil.copy(f"data/{name}", f"{OUT}/{name}")

with open(f"{OUT}/README.md", "w") as f:
    f.write("Subset of the LBNL Fault Detection and Diagnostics Data Sets - Chiller Plant\n"
            "(Lawrence Berkeley National Laboratory), used by the online dashboard:\n"
            "test days only, steady chiller operation, the columns used by the detectors.\n"
            "Made by src/make_demo_data.py.\n")
print(f"\nSaved {len(catalog)} files to {OUT}/ ({total:.1f} MB)")
