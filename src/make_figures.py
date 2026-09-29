import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Figures for the final report, saved to reports/figures/ as PNG (200 dpi).
# Run after train_model.py, evaluate_faults.py, experiment.py and
# export_model.py, which write the files read here.

OUT = "reports/figures"
os.makedirs(OUT, exist_ok=True)

# Colors: fixed categorical order (validated for color-vision deficiency),
# neutral ink for all text, recessive grid.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.edgecolor": GRID,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False,
})


def save(fig, name, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(f"{OUT}/{name}.png", dpi=200)
    plt.close(fig)
    print(f"Saved {OUT}/{name}.png")


# --- Figure 1: training curve ---
history = np.load("data/processed/loss_history.npy")      # (epochs, 2): train, val
epochs = np.arange(1, len(history) + 1)
best = int(np.argmin(history[:, 1])) + 1

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(epochs, history[:, 0], color=BLUE, lw=2, label="Train")
ax.plot(epochs, history[:, 1], color=ORANGE, lw=2, label="Validation")
ax.set_yscale("log")
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3g"))
ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.3g"))
ax.tick_params(axis="y", which="minor", labelsize=8)
ax.axvline(best, color=MUTED, lw=1, ls="--")
ax.annotate(f"best epoch {best}\nval loss {history[best - 1, 1]:.4f}",
            xy=(best, history[best - 1, 1]), xytext=(-10, 30), textcoords="offset points",
            ha="right", color=INK_2, fontsize=9)
ax.set_xlabel("Epoch")
ax.set_ylabel("Reconstruction loss (MSE, log scale)")
ax.set_title("Training curve - autoencoder (16, 6)")
ax.legend(loc="upper right")
save(fig, "1_training_curve")


# --- Figure 2: detection per fault file, row vs 1-hour window ---
res = pd.read_csv("data/fault_results.csv", dtype={"severity": str})
res = res.sort_values(["fault_type", "severity"]).reset_index(drop=True)
labels = [f"{t}  {s}" for t, s in zip(res["fault_type"], res["severity"])]
y = np.arange(len(res))[::-1]
h = 0.38

fig, ax = plt.subplots(figsize=(8, 0.34 * len(res) + 1.4))
ax.barh(y + h / 2, res["window_detection_pct"], height=h - 0.04, color=BLUE, label="1-hour window")
ax.barh(y - h / 2, res["row_detection_pct"], height=h - 0.04, color=ORANGE, label="Single row")
for yi, v in zip(y, res["window_detection_pct"]):
    ax.text(v + 1, yi + h / 2, f"{v:.0f}%", va="center", fontsize=7.5, color=INK_2)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlim(0, 108)
ax.set_xlabel("Detected (% of rows or windows, test days)")
ax.set_title("Fault detection per fault file")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower right", fontsize=9)
save(fig, "2_detection_per_fault")


# --- Figure 3: mean detection per fault type (1-hour window) ---
by_type = res.groupby("fault_type")["window_detection_pct"].agg(["mean", "min", "max"])
by_type = by_type.sort_values("mean")
yt = np.arange(len(by_type))

fig, ax = plt.subplots(figsize=(8, 3.8))
ax.barh(yt, by_type["mean"], height=0.6, color=BLUE)
ax.errorbar(by_type["mean"], yt, xerr=[by_type["mean"] - by_type["min"], by_type["max"] - by_type["mean"]],
            fmt="none", ecolor=INK_2, elinewidth=1, capsize=3)
for yi, v, top in zip(yt, by_type["mean"], by_type["max"]):
    ax.text(max(v, top) + 1.5, yi, f"{v:.0f}%", va="center", fontsize=9, color=INK)
ax.set_yticks(yt)
ax.set_yticklabels(by_type.index, fontsize=9)
ax.set_xlim(0, 110)
ax.set_xlabel("Hourly detection (%): bar = mean over severities, whiskers = min-max")
ax.set_title("Detection per fault type")
ax.grid(axis="y", visible=False)
save(fig, "3_detection_per_type")


# --- Figure 4: how the method improved, and what was rejected ---
# Numbers from the runs documented in the commit history:
#   test days: row (8,4) 24.5%, hour (8,4) 58.4%, hour (16,6) 72.8%
#   val days (used for decisions): final 72.5%, + control sensors 64.3%,
#   + delta features 69.2%
steps = pd.DataFrame({
    "stage": ["Row score\nmodel (8,4)", "Hourly score\nmodel (8,4)", "Hourly score\nmodel (16,6)"],
    "value": [24.5, 58.4, 72.8],
})
tried = pd.DataFrame({
    "stage": ["Final model", "+ 3 controller\nsignals", "+ 4 delta\nfeatures"],
    "value": [72.5, 64.3, 69.2],
})

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, df, title, colors in [
    (axes[0], steps, "Improvements (test days)", [MUTED, MUTED, BLUE]),
    (axes[1], tried, "Rejected changes (validation days)", [BLUE, ORANGE, ORANGE]),
]:
    ax.bar(df["stage"], df["value"], width=0.6, color=colors)
    for i, v in enumerate(df["value"]):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9, color=INK)
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Mean detection over 23 fault files (%)")
save(fig, "4_method_progression")


# --- Figure 5: experiment - model size x window length ---
exp = pd.read_csv("data/experiment_results.csv")
exp["model"] = "(" + exp["hidden"].astype(str) + "," + exp["bottleneck"].astype(str) + ")\n" \
               + exp["params"].astype(str) + " p"
order = exp.drop_duplicates(["hidden", "bottleneck"])["model"].tolist()
windows = {60: ("1 hour", BLUE), 180: ("3 hours", ORANGE), 1440: ("1 day", AQUA)}

fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
for ax, score, title in [(axes[0], "max_z", "Max-z score (per sensor)"),
                         (axes[1], "mahalanobis", "Mahalanobis score (all sensors)")]:
    ends = []
    for w, (name, color) in windows.items():
        d = exp[(exp["score"] == score) & (exp["window_min"] == w)].set_index("model").loc[order]
        x = np.arange(len(order))
        ax.plot(x, d["detection_val_days_pct"], color=color, lw=2, marker="o", ms=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
        ends.append([d["detection_val_days_pct"].iloc[-1], name])
    # End labels: push apart labels that would overlap (min 2 points apart)
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 2.0)
    for y_end, name in ends:
        ax.text(len(order) - 1 + 0.12, y_end, name, color=INK_2, fontsize=8, va="center")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, fontsize=8)
    ax.set_xlim(-0.3, len(order) - 0.3)
    ax.set_title(title)
    ax.set_xlabel("Model (hidden, bottleneck), parameters")
axes[0].set_ylabel("Mean detection, validation days (%)")
handles, names = axes[0].get_legend_handles_labels()
fig.legend(handles, names, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0))
save(fig, "5_experiment_model_window", rect=(0, 0.07, 1, 1))


# --- Figure 6: edge export - one small panel per metric (different units) ---
edge = pd.read_csv("data/edge_results.csv", index_col=0)
versions = edge.index.tolist()
colors = [MUTED, BLUE, ORANGE][:len(versions)]
metrics = [("size_kb", "Size (KB)"), ("latency_median_us", "Latency per row (µs, median)"),
           ("energy_per_inference_mJ", "Energy per inference (mJ, est.)"),
           ("detection_pct", "Hourly detection (%)")]

fig, axes = plt.subplots(1, 4, figsize=(11, 3.4))
for ax, (col, title) in zip(axes, metrics):
    ax.bar(range(len(versions)), edge[col], width=0.6, color=colors)
    for i, v in enumerate(edge[col]):
        ax.text(i, v, f"{v:.1f}" if v >= 1 else f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(range(len(versions)))
    ax.set_xticklabels([v.replace(" ", "\n") for v in versions], fontsize=8)
    ax.set_title(title, fontsize=9.5)
    ax.set_ylim(0, edge[col].max() * 1.18)
    ax.grid(axis="x", visible=False)
save(fig, "6_edge_export")

print(f"\nAll figures saved to {OUT}/")
