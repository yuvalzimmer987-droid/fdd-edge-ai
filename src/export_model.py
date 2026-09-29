import os
import time
import numpy as np
import pandas as pd
import joblib
import torch
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from codecarbon import EmissionsTracker
from model import Autoencoder                 # defined in src/model.py
from preprocessing import DATA_DIR, load_plant_file, day_split, steady_state_mask

# Edge export: PyTorch -> ONNX (float32) -> ONNX int8, and for each version:
#   1. size on disk
#   2. latency for ONE row (on the edge device, one new reading per minute)
#   3. energy per inference (CodeCarbon estimate)
#   4. fault detection, recomputed end-to-end with that version, to check
#      that exporting / quantizing does not hurt the results
# The detection method is the same as evaluate_faults.py: 1-hour windows,
# max-z score, thresholds from normal validation data, scored on test days.

FP32_PATH = "models/autoencoder.onnx"
INT8_PATH = "models/autoencoder_int8.onnx"
LATENCY_RUNS = 5000
ENERGY_RUNS = 50000
PERCENTILE = 99
WINDOW_MINUTES = 60
MIN_ROWS_PER_WINDOW = 30

# --- Step 1: load the trained PyTorch model ---
scaler = joblib.load("models/scaler.joblib")
sensors = list(scaler.feature_names_in_)
n = len(sensors)

model = Autoencoder(n)
model.load_state_dict(torch.load("models/autoencoder.pt"))
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"PyTorch model: {n} inputs, {n_params} parameters")

# --- Step 2: export to ONNX (float32), with a variable batch size ---
torch.onnx.export(
    model, torch.zeros(1, n), FP32_PATH,
    input_names=["input"], output_names=["reconstruction"],
    dynamic_axes={"input": {0: "batch"}, "reconstruction": {0: "batch"}},
    opset_version=17, dynamo=False,
)
print(f"Exported {FP32_PATH}")

# --- Step 3: quantize to int8 (weights stored as 8-bit integers) ---
quantize_dynamic(FP32_PATH, INT8_PATH, weight_type=QuantType.QInt8)
print(f"Quantized {INT8_PATH}")


# --- One "predict" function per version, all with the same interface ---
def torch_predict(x):
    with torch.no_grad():
        return model(torch.from_numpy(x)).numpy()


def onnx_predictor(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1           # an edge device has few cores
    session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
    return lambda x: session.run(None, {"input": x})[0]


def one_row_at_a_time(predict, x):
    """Run the model row by row, as on the edge device (one reading per minute).

    This matters for int8: dynamic quantization picks its value range from the
    batch it is given, so a big mixed batch would give it a coarser range than
    a single row gets in real use.
    """
    return np.concatenate([predict(x[i:i + 1]) for i in range(len(x))])


torch.set_num_threads(1)
versions = {
    "PyTorch fp32": (torch_predict, "models/autoencoder.pt"),
    "ONNX fp32":    (onnx_predictor(FP32_PATH), FP32_PATH),
    "ONNX int8":    (onnx_predictor(INT8_PATH), INT8_PATH),
}

# --- Step 4: check the ONNX outputs match PyTorch ---
val = np.load("data/processed/val.npy").astype(np.float32)
sample = val[:2000]
reference = torch_predict(sample)
print("\nDifference from PyTorch output, row by row (scaled units, 2000 rows):")
for name, (predict, _) in versions.items():
    diff = np.abs(one_row_at_a_time(predict, sample) - reference)
    print(f"   {name:13s} mean {diff.mean():.6f}   max {diff.max():.6f}")

# --- Step 5: size, latency and energy ---
print("\n--- Size, latency (1 row) and energy ---")
one_row = val[:1]
perf = {}
for name, (predict, path) in versions.items():
    for _ in range(200):                       # warm-up
        predict(one_row)
    times = []
    for i in range(LATENCY_RUNS):
        row = val[i % len(val)][None, :]
        start = time.perf_counter()
        predict(row)
        times.append(time.perf_counter() - start)
    times = np.array(times) * 1e6              # microseconds

    tracker = EmissionsTracker(project_name=f"inference_{name}", log_level="error",
                               save_to_file=False)
    tracker.start()
    for i in range(ENERGY_RUNS):
        predict(val[i % len(val)][None, :])
    tracker.stop()
    energy_j = tracker.final_emissions_data.energy_consumed * 3.6e6 / ENERGY_RUNS

    perf[name] = {
        "size_kb": os.path.getsize(path) / 1024,
        "latency_median_us": np.median(times),
        "latency_p99_us": np.percentile(times, 99),
        "energy_per_inference_mJ": energy_j * 1000,
    }
    print(f"   {name:13s} size {perf[name]['size_kb']:6.1f} KB   "
          f"latency median {perf[name]['latency_median_us']:7.1f} us  "
          f"p99 {perf[name]['latency_p99_us']:7.1f} us   "
          f"energy {perf[name]['energy_per_inference_mJ']:.4f} mJ/inference")


# --- Step 6: fault detection with each version ---
def window_means(idx, resid):
    df = pd.DataFrame(resid)
    window = idx // WINDOW_MINUTES
    counts = df.groupby(window).size()
    return df.groupby(window).mean()[counts >= MIN_ROWS_PER_WINDOW].to_numpy()


print("\nLoading fault files (test days)...")
normal = {
    "val":  (np.load("data/processed/val_idx.npy"), val),
    "test": (np.load("data/processed/test_normal_idx.npy"),
             np.load("data/processed/test_normal.npy").astype(np.float32)),
}
catalog = pd.read_csv("data/file_catalog.csv", dtype={"severity": str})
faults = []
for _, row in catalog[catalog["label"] == "fault"].iterrows():
    df = load_plant_file(f"{DATA_DIR}/{row['filename']}")
    keep, _, _, _ = steady_state_mask(df)
    mask = keep & (day_split(len(df)) == "test")
    faults.append((np.flatnonzero(mask),
                   scaler.transform(df.loc[mask, sensors]).astype(np.float32)))

print("\n--- Fault detection per version (1-hour windows, test days) ---")
for name, (predict, _) in versions.items():
    resid = {p: (idx, one_row_at_a_time(predict, x) - x) for p, (idx, x) in normal.items()}
    val_win = window_means(*resid["val"])
    center, spread = val_win.mean(axis=0), val_win.std(axis=0)
    score = lambda w: np.abs((w - center) / spread).max(axis=1)
    threshold = np.percentile(score(val_win), PERCENTILE)

    false_alarm = (score(window_means(*resid["test"])) > threshold).mean() * 100
    detection = np.mean([(score(window_means(idx, one_row_at_a_time(predict, x) - x))
                          > threshold).mean() * 100 for idx, x in faults])
    perf[name]["false_alarm_pct"] = false_alarm
    perf[name]["detection_pct"] = detection
    print(f"   {name:13s} detection {detection:5.1f}%   false alarms {false_alarm:5.2f}%")

# --- Summary ---
summary = pd.DataFrame(perf).T
summary.to_csv("data/edge_results.csv")
print("\n=== Summary ===")
print(summary.round(4).to_string())
print("\nSaved to data/edge_results.csv")
