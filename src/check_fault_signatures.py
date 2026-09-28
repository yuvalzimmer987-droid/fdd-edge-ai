import numpy as np
import pandas as pd
import joblib
from preprocessing import DATA_DIR, day_split, steady_state_mask

# Which of ALL the columns in the files does each fault change?
#
# The fault files are simulations of the same year, with the same weather, as
# the fault-free file. So row N of a fault file can be compared directly with
# row N of the fault-free file: any difference is caused by the fault.
#
# This shows whether the faults the model misses (fouling 095, positive
# pressure bias) are visible in sensors the model does NOT use yet.
#
# Only train + validation days are used here, so that choosing extra sensors
# from this analysis does not touch the test days used for the final results.

N_TOP = 8   # columns to show per fault file

scaler = joblib.load("models/scaler.joblib")
model_sensors = set(scaler.feature_names_in_)


def load_all_columns(path):
    """Load every numeric column, with the same OA_TEMP / OA_TEMP_WB fix."""
    df = pd.read_csv(path)
    df = df.rename(columns={"OA_TEMP": "OA_TEMP_WB", "OA_TEMP_WB": "OA_TEMP"})
    return df.select_dtypes("number").astype("float32")


print("Loading fault-free file (all columns)...")
normal = load_all_columns(f"{DATA_DIR}/ChillerPlant.csv")
normal_keep, _, _, _ = steady_state_mask(normal)
not_test = day_split(len(normal)) != "test"
columns = list(normal.columns)
print(f"{len(columns)} numeric columns, {len(model_sensors)} used by the model.\n")

catalog = pd.read_csv("data/file_catalog.csv", dtype={"severity": str})
rows = []

for _, row in catalog[catalog["label"] == "fault"].iterrows():
    fault = load_all_columns(f"{DATA_DIR}/{row['filename']}")
    if len(fault) != len(normal) or list(fault.columns) != columns:
        print(f"SKIP {row['filename']}: different length or columns than the fault-free file")
        continue

    # Rows where BOTH files are in steady state (the fault can change when the
    # chiller runs), on train + validation days only
    fault_keep, _, _, _ = steady_state_mask(fault)
    mask = normal_keep & fault_keep & not_test

    n = normal.loc[mask]
    f = fault.loc[mask]
    diff = f - n
    spread = n.std()

    # Effect size, in units of the column's normal spread (0.5 = half a std):
    # - shift:  |average difference|. Catches a steady offset (e.g. a bias).
    # - effect: root-mean-square difference. Catches ANY change, including
    #   oscillations (e.g. a badly tuned controller) that average out to ~0.
    std = spread.replace(0, np.nan)
    shift = (diff.mean() / std).abs()
    effect = np.sqrt((diff ** 2).mean()) / std
    # A column that is constant when normal but changes under the fault
    effect[(spread == 0) & (diff.abs().max() > 0)] = np.inf

    severity = row["severity"] if isinstance(row["severity"], str) else "-"
    name = f"{row['fault_type']} {severity}"
    print(f"=== {name}  ({mask.sum()} rows compared) ===")
    for col in effect.sort_values(ascending=False).head(N_TOP).index:
        used = "model" if col in model_sensors else "NOT USED"
        print(f"   {col:28s} effect {effect[col]:7.2f}   shift {shift[col]:7.2f} "
              f"(mean {diff[col].mean():+10.3f})   {used}")
    print()

    for col in columns:
        rows.append({"fault": name, "column": col, "effect": effect[col], "shift": shift[col],
                     "mean_diff": diff[col].mean(), "in_model": col in model_sensors})

result = pd.DataFrame(rows)
result.to_csv("data/fault_signatures.csv", index=False)

# --- Summary 1: the faults the model misses ---
# For each, the strongest column in the model vs the strongest one outside it.
print("=== Strongest effect inside vs outside the model, per fault ===")
print(f"{'fault':45s} {'best in model':>30s} {'best NOT in model':>34s}")
for name, g in result.groupby("fault", sort=False):
    inside = g[g["in_model"]].sort_values("effect", ascending=False).iloc[0]
    outside = g[~g["in_model"]].sort_values("effect", ascending=False)
    out_txt = (f"{outside.iloc[0]['column']} {outside.iloc[0]['effect']:.2f}"
               if len(outside) else "-")
    print(f"{name:45s} {inside['column'] + ' ' + format(inside['effect'], '.2f'):>30s} {out_txt:>34s}")

# --- Summary 2: columns not in the model that react to many faults ---
# Good candidates to add: sensors that show many different faults, not one.
print("\n=== Columns NOT in the model, by how many fault files move them by > 0.5 std ===")
outside = result[~result["in_model"]]
summary = outside.groupby("column").agg(
    faults_over_0_5=("effect", lambda e: int((e > 0.5).sum())),
    max_effect=("effect", "max"),
).sort_values(["faults_over_0_5", "max_effect"], ascending=False)
print(summary.head(20).round(2).to_string())

print("\nFull table saved to data/fault_signatures.csv")
