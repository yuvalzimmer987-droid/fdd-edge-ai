import os
import time
import numpy as np
import pandas as pd
import torch
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from codecarbon import EmissionsTracker
from detector import Detector, load_torch_models, torch_predictor
from preprocessing import DATA_DIR, load_plant_file, day_split, steady_state_mask

# Edge export of the full detector (main model + pump detector):
# PyTorch -> ONNX (float32) -> ONNX int8, and for each version:
#   1. size on disk (both models)
#   2. latency for ONE minute of data (one row through both models - on the
#      edge device a new reading arrives every minute)
#   3. energy per minute of data (CodeCarbon estimate)
#   4. fault detection, recomputed end-to-end with that version: thresholds
#      re-fitted on normal validation days at the same false-alarm budget as
#      the saved detector, scored on test days. This checks that exporting /
#      quantizing does not hurt the results.
#
# Needs: train_model.py and subsystems.py --hybrid run first.
# Run:   python src/export_model.py

LATENCY_RUNS = 5000
ENERGY_RUNS = 50000

# --- Step 1: the trained PyTorch models ---
models, scalers = load_torch_models()
budget = Detector.from_saved().budget
for name, m in models.items():
    print(f"{name:6s}: {len(scalers[name].feature_names_in_)} inputs, "
          f"{sum(p.numel() for p in m.parameters())} parameters")
print(f"False-alarm budget: {budget}%")

# --- Step 2: export to ONNX (float32) and quantize to int8 ---
paths = {"fp32": {}, "int8": {}}
for name, m in models.items():
    n = len(scalers[name].feature_names_in_)
    paths["fp32"][name] = f"models/{name}.onnx"
    paths["int8"][name] = f"models/{name}_int8.onnx"
    torch.onnx.export(
        m, torch.zeros(1, n), paths["fp32"][name],
        input_names=["input"], output_names=["reconstruction"],
        dynamic_axes={"input": {0: "batch"}, "reconstruction": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    quantize_dynamic(paths["fp32"][name], paths["int8"][name], weight_type=QuantType.QInt8)
print("Exported " + ", ".join(p for v in paths.values() for p in v.values()))


def onnx_predictor(path):
    """Run the model row by row, as on the edge device (one reading per minute).

    This matters for int8: dynamic quantization picks its value range from the
    batch it is given, so a big mixed batch would give it a coarser range than
    a single row gets in real use.
    """
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1           # an edge device has few cores
    session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
    return lambda x: np.concatenate([session.run(None, {"input": x[i:i + 1]})[0]
                                     for i in range(len(x))])


torch.set_num_threads(1)
versions = {
    "PyTorch fp32": ({n: torch_predictor(m) for n, m in models.items()},
                     [f"models/{f}" for f in ["autoencoder.pt", "subsys_pumps.pt"]]),
    "ONNX fp32": ({n: onnx_predictor(p) for n, p in paths["fp32"].items()}, list(paths["fp32"].values())),
    "ONNX int8": ({n: onnx_predictor(p) for n, p in paths["int8"].items()}, list(paths["int8"].values())),
}

# --- Step 3: data - the fault-free file (validation + test days) and every fault file (test days) ---
columns = Detector(versions["PyTorch fp32"][0], scalers).load_columns()
normal = load_plant_file(f"{DATA_DIR}/ChillerPlant.csv", columns)
keep, _, _, _ = steady_state_mask(normal)
split = day_split(len(normal))
normal_rows = {p: np.flatnonzero(keep & (split == p)) for p in ["val", "test"]}

print("\nLoading fault files (test days)...")
catalog = pd.read_csv("data/file_catalog.csv", dtype={"severity": str})
faults = []
for _, row in catalog[catalog["label"] == "fault"].iterrows():
    df = load_plant_file(f"{DATA_DIR}/{row['filename']}", columns)
    f_keep, _, _, _ = steady_state_mask(df)
    idx = np.flatnonzero(f_keep & (day_split(len(df)) == "test"))
    faults.append((df.iloc[idx], idx))

# --- Step 4: check the ONNX outputs match PyTorch ---
print("\nDifference from PyTorch output, row by row (scaled units, 2000 rows):")
sample = normal.iloc[normal_rows["val"][:2000]]
for version, (predictors, _) in versions.items():
    diffs = []
    for name in models:
        x = scalers[name].transform(sample[scalers[name].feature_names_in_]).astype(np.float32)
        diffs.append(np.abs(predictors[name](x) - versions["PyTorch fp32"][0][name](x)))
    print(f"   {version:13s} " + "   ".join(f"{n} mean {d.mean():.6f} max {d.max():.6f}"
                                         for n, d in zip(models, diffs)))

# --- Step 5: size, latency and energy per minute of data ---
print("\n--- Size, latency and energy (one minute of data = one row through both models) ---")
rows = {name: scalers[name].transform(sample[scalers[name].feature_names_in_]).astype(np.float32)
        for name in models}
perf = {}
for version, (predictors, files) in versions.items():
    def one_minute(i):
        for name, predict in predictors.items():
            predict(rows[name][i % len(sample)][None, :])

    for i in range(200):                       # warm-up
        one_minute(i)
    times = []
    for i in range(LATENCY_RUNS):
        start = time.perf_counter()
        one_minute(i)
        times.append(time.perf_counter() - start)
    times = np.array(times) * 1e6              # microseconds

    tracker = EmissionsTracker(project_name=f"inference_{version}", log_level="error",
                               save_to_file=False)
    tracker.start()
    for i in range(ENERGY_RUNS):
        one_minute(i)
    tracker.stop()
    energy_j = tracker.final_emissions_data.energy_consumed * 3.6e6 / ENERGY_RUNS

    perf[version] = {
        "size_kb": sum(os.path.getsize(f) for f in files) / 1024,
        "latency_median_us": np.median(times),
        "latency_p99_us": np.percentile(times, 99),
        "energy_per_minute_mJ": energy_j * 1000,
    }
    print(f"   {version:13s} size {perf[version]['size_kb']:6.1f} KB   "
          f"latency median {perf[version]['latency_median_us']:7.1f} us  "
          f"p99 {perf[version]['latency_p99_us']:7.1f} us   "
          f"energy {perf[version]['energy_per_minute_mJ']:.4f} mJ/minute")

# --- Step 6: fault detection with each version ---
print(f"\n--- Fault detection per version (1-hour windows, test days, {budget}% budget) ---")
for version, (predictors, _) in versions.items():
    detector = Detector(predictors, scalers)
    val_idx = normal_rows["val"]
    detector.calibrate(normal.iloc[val_idx], val_idx, budget)
    test_idx = normal_rows["test"]
    false_alarm = detector.hourly(normal.iloc[test_idx], test_idx)["alarm"].mean() * 100
    detection = np.mean([detector.hourly(frame, idx)["alarm"].mean() * 100 for frame, idx in faults])
    perf[version]["false_alarm_pct"] = false_alarm
    perf[version]["detection_pct"] = detection
    print(f"   {version:13s} detection {detection:5.1f}%   false alarms {false_alarm:5.2f}%")

# --- Summary ---
summary = pd.DataFrame(perf).T
summary.to_csv("data/edge_results.csv")
print("\n=== Summary ===")
print(summary.round(4).to_string())
print("\nSaved to data/edge_results.csv")
