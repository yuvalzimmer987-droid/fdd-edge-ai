import sys
import json
import numpy as np
import pandas as pd
from detector import Detector
from preprocessing import DATA_DIR, load_plant_file, day_split, steady_state_mask, ROWS_PER_DAY
from signatures import FAULTS, SIGNATURES_FILE, fault_class, signature_vectors

# Learn the typical pattern (signature) of each fault class and check how often
# the most similar signature names the right fault.
#
# Signatures come from the alarm hours of the VALIDATION days of every fault
# file; the check runs on the alarm hours of the TEST days, as for detection.
# Only hours the detector flags are used: a signature is matched only when an
# alarm is raised.
#
# Run: python src/fault_signatures.py          (needs the full data folder)
#      python src/fault_signatures.py --demo   (code check on demo_data/: test
#                                               days split in two - numbers
#                                               are NOT for the report)

DEMO = "--demo" in sys.argv
detector = Detector.from_saved()
catalog = pd.read_csv("demo_data/file_catalog.csv" if DEMO else "data/file_catalog.csv",
                      dtype={"severity": str})


def parts_of(filename):
    """(frame, idx) of the validation and test days of one file."""
    if DEMO:
        frame = pd.read_parquet(f"demo_data/{filename.replace('.csv', '.parquet')}")
        idx = frame.index.to_numpy()
        even = (idx // ROWS_PER_DAY) % 2 == 0
        return {"val": (frame[even], idx[even]), "test": (frame[~even], idx[~even])}
    df = load_plant_file(f"{DATA_DIR}/{filename}", detector.load_columns())
    keep, _, _, _ = steady_state_mask(df)
    split = day_split(len(df))
    out = {}
    for part in ["val", "test"]:
        idx = np.flatnonzero(keep & (split == part))
        out[part] = (df.iloc[idx], idx)
    return out


# --- Step 1: the signed pattern of every alarm hour, per fault file ---
patterns = {"val": [], "test": []}
for _, row in catalog[catalog["label"] == "fault"].iterrows():
    cls = fault_class(row["fault_type"], row["severity"])
    for part, (frame, idx) in parts_of(row["filename"]).items():
        signed = detector.signed_z_scores(frame, idx)
        hours = detector.hourly(frame, idx, {n: z.abs() for n, z in signed.items()})
        vec = signature_vectors(signed)[hours["alarm"]]
        patterns[part].append(vec.assign(cls=cls, file=row["filename"]))
val = pd.concat(patterns["val"])
test = pd.concat(patterns["test"])
features = [c for c in val.columns if c not in ("cls", "file")]
print(f"Alarm hours: {len(val)} on validation days, {len(test)} on test days")

# --- Step 2: one signature per class = mean direction of its validation alarm hours ---
classes = {}
for cls, group in val.groupby("cls"):
    centroid = group[features].fillna(0).mean()
    centroid = centroid / np.sqrt((centroid ** 2).sum())
    classes[cls] = {"centroid": centroid.round(5).tolist(), "val_hours": int(len(group))}
names = list(classes)
C = pd.DataFrame({k: v["centroid"] for k, v in classes.items()}, index=features)

# --- Step 3: check on test days - is the most similar class the right one? ---
sim = test[features].fillna(0) @ C
ranked = np.argsort(-sim.to_numpy(), axis=1)
true = test["cls"].to_numpy()
top1 = np.array(names)[ranked[:, 0]]
in_top3 = np.array([t in np.array(names)[r[:3]] for t, r in zip(true, ranked)])
type_of = lambda c: c.replace("_pos", "").replace("_neg", "")
same_type = np.array([type_of(a) == type_of(b) for a, b in zip(true, top1)])

per_class = (pd.DataFrame({"cls": true, "ok": top1 == true, "top3": in_top3})
             .groupby("cls").agg(hours=("ok", "size"), top1=("ok", "mean"), top3=("top3", "mean")))
per_class[["top1", "top3"]] = (per_class[["top1", "top3"]] * 100).round(1)
accuracy = {
    "top1_pct": round(float((top1 == true).mean() * 100), 1),
    "top3_pct": round(float(in_top3.mean() * 100), 1),
    "type_top1_pct": round(float(same_type.mean() * 100), 1),
    "class_balanced_top1_pct": round(float(per_class["top1"].mean()), 1),
    "test_alarm_hours": int(len(test)),
}

print("\n=== Most similar signature = the right fault? (alarm hours, "
      f"{'DEMO CHECK - not for the report' if DEMO else 'test days'}) ===")
print(per_class.rename(index=lambda c: FAULTS[c]["name"]).to_string())
print(f"\nAll alarm hours: top-1 {accuracy['top1_pct']}%, top-3 {accuracy['top3_pct']}%, "
      f"right fault type (ignoring the sign of a bias) {accuracy['type_top1_pct']}%, "
      f"mean over classes {accuracy['class_balanced_top1_pct']}%")
short = {c: c.replace("secondary_chilled_water_pressure", "press").replace("coolingtower", "ct")
         .replace("chiller", "chl").replace("_bias", "_b").replace("_fouling", "_foul") for c in names}
confusion = pd.crosstab(pd.Series(true, name="true").map(short), pd.Series(top1, name="predicted").map(short))
print("\nConfusion (rows = true class, columns = most similar class, alarm hours):")
print(confusion.to_string())

if DEMO:
    print("\n--demo: nothing saved.")
else:
    with open(SIGNATURES_FILE, "w") as f:
        json.dump({"features": features, "classes": classes, "accuracy": accuracy,
                   "per_class": per_class.reset_index().to_dict(orient="records")}, f, indent=1)
    print(f"\nSaved {SIGNATURES_FILE}")
