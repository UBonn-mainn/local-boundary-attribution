from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Load data
# =========================
ROOT = Path("/Users/nguyennhatmai/Documents/study/UBonn/WiSe2526/LabDMAI/local-boundary-attribution/results/synthetic_experiments")

dfs = []
for report_path in ROOT.rglob("report_summary.csv"):
    dataset = report_path.parent.name
    if "spirals" in dataset: continue
    df_raw = pd.read_csv(report_path)
    df_raw["dataset"] = dataset
    dfs.append(df_raw)

df = pd.concat(dfs, ignore_index=True)

# keep only IG
df = df[df["method"] == "ig"].copy()

# prettier baseline labels
baseline_map = {
    "zero": "Zero",
    "boundary": "Boundary"
}
df["baseline"] = df["baseline"].map(baseline_map)

dataset_order = [
    "2d_linear_blobs", "2d_moons", "2d_3class", "2d_10class",
    "8d_linear_blobs", "8d_moons", "8d_3class", "8d_10class",
    "15d_linear_blobs", "15d_moons", "15d_3class"
]

df["dataset"] = pd.Categorical(df["dataset"], categories=dataset_order, ordered=True)
df = df.sort_values("dataset")

x = np.arange(len(dataset_order))
bar_width = 0.35


def pivot_metric(metric):
    p = df.pivot(index="dataset", columns="baseline", values=metric)
    return p.reindex(dataset_order)


# =========================
# Metrics to plot
# =========================
metrics = [
    ("fidelity_del_auc_mean", "Deletion AUC"),
    ("fidelity_ins_auc_mean", "Insertion AUC"),
    ("cosine_truegrad_mean", "Cosine to True Gradient"),
    ("boundary_alignment_mean", "Boundary Alignment"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

for ax, (metric, title) in zip(axes.flatten(), metrics):

    pivot = pivot_metric(metric)

    bar1 = ax.bar(x - bar_width/2, pivot["Zero"], bar_width)
    bar2 = ax.bar(x + bar_width/2, pivot["Boundary"], bar_width)

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_order, rotation=45, ha="right")

    ax.set_title(title)
    ax.set_ylabel("Score")

# shared legend
fig.legend(
    [bar1, bar2],
    ["Zero baseline", "Boundary baseline"],
    loc="upper center",
    ncol=2,
    fontsize=11
)

# fig.suptitle("Integrated Gradients Attribution Metrics Across All Datasets", fontsize=15)

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig("./attribution/ig_metrics_all_datasets.png", dpi=300, bbox_inches="tight")
plt.show()