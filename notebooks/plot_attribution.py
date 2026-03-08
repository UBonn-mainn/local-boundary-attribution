from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Paths
# ============================================================
ROOT = Path("/Users/nguyennhatmai/Documents/study/UBonn/WiSe2526/LabDMAI/local-boundary-attribution/results/synthetic_experiments")
OUTDIR = ROOT / "plots_summary_report"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Shared loader
# ============================================================
def load_all_csvs(root: Path, filename: str, skip_keyword: str = "spirals") -> pd.DataFrame:
    dfs = []
    for csv_path in root.rglob(filename):
        dataset = csv_path.parent.name
        if skip_keyword and skip_keyword in dataset:
            continue

        df_raw = pd.read_csv(csv_path)
        df_raw["dataset"] = dataset
        dfs.append(df_raw)

    if not dfs:
        raise FileNotFoundError(f"No {filename} files found under {root}")

    return pd.concat(dfs, ignore_index=True)

# ============================================================
# Load data
# ============================================================
summary_df = load_all_csvs(ROOT, "report_summary.csv")
disagreement_df = load_all_csvs(ROOT, "report_pairwise_disagreement.csv")

# ============================================================
# Orders / labels
# ============================================================
dataset_order = [
    "2d_linear_blobs", "2d_moons", "2d_3class", "2d_10class",
    "8d_linear_blobs", "8d_moons", "8d_3class", "8d_10class",
    "15d_linear_blobs", "15d_moons", "15d_3class"
]

dataset_label_map = {
    "2d_linear_blobs": "2d_linear_blobs",
    "2d_moons": "2d_moons",
    "2d_3class": "2d_3class",
    "2d_10class": "2d_10class",
    "8d_linear_blobs": "8d_linear_blobs",
    "8d_moons": "8d_moons",
    "8d_3class": "8d_3class",
    "8d_10class": "8d_10class",
    "15d_linear_blobs": "15d_linear_blobs",
    "15d_moons": "15d_moons",
    "15d_3class": "15d_3class",
}

method_map = {"ig": "IG", "lime": "LIME", "ks": "KS"}
baseline_map = {
    "zero": "Zero",
    "boundary": "BC",          # backward compatibility
    "boundarycrawler": "BC",
    "bc": "BC",                # backward compatibility
    "gs": "GS",
}

group_order = [
    "IG_Zero", "IG_BC", "IG_GS",
    "LIME_Zero", "LIME_BC", "LIME_GS",
    "KS_Zero", "KS_BC", "KS_GS",
]

group_label_map = {
    "IG_Zero": "IG/\nZero",
    "IG_BC": "IG/\nBC",
    "IG_GS": "IG/\nGS",
    "LIME_Zero": "LIME/\nZero",
    "LIME_BC": "LIME/\nBC",
    "LIME_GS": "LIME/\nGS",
    "KS_Zero": "KS/\nZero",
    "KS_BC": "KS/\nBC",
    "KS_GS": "KS/\nGS",
}

palette = {

    # IG (blue family)
    "IG_Zero": "#A6C8E0",
    "IG_BC": "#4C78A8",
    "IG_GS": "#1F4E79",

    # LIME (green family)
    "LIME_Zero": "#A1D99B",
    "LIME_BC": "#54A24B",
    "LIME_GS": "#1B7837",

    # KS (purple family)
    "KS_Zero": "#D4B9DA",
    "KS_BC": "#B279A2",
    "KS_GS": "#6A3D9A",
}

# ============================================================
# Prepare summary data
# ============================================================
summary_df["method"] = summary_df["method"].map(method_map).fillna(summary_df["method"])
summary_df["baseline"] = summary_df["baseline"].map(baseline_map).fillna(summary_df["baseline"])
summary_df["group"] = summary_df["method"] + "_" + summary_df["baseline"]

summary_df["dataset"] = pd.Categorical(summary_df["dataset"], categories=dataset_order, ordered=True)
summary_df = summary_df.sort_values(["dataset", "method", "baseline"]).copy()

# keep only supported groups
summary_df = summary_df[summary_df["group"].isin(group_order)].copy()

# ============================================================
# Helpers
# ============================================================
def prepare_metric_groups(df: pd.DataFrame, metric: str):
    data, labels, colors = [], [], []
    for group in group_order:
        vals = df.loc[df["group"] == group, metric].dropna().astype(float).values
        data.append(vals)
        labels.append(group_label_map[group])
        colors.append(palette[group])
    return data, labels, colors

def colored_boxplot(ax, data, labels, colors, ylabel, title):
    bp = ax.boxplot(
        data,
        patch_artist=True,
        labels=labels,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
        boxprops={"linewidth": 1.0},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis="x", rotation=35, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

def columnwise_normalize(df: pd.DataFrame):
    out = df.copy()
    for col in out.columns:
        vals = out[col].astype(float)
        if vals.isna().all():
            out[col] = 0.0
        else:
            mn, mx = vals.min(), vals.max()
            if np.isclose(mn, mx):
                out[col] = 0.0
            else:
                out[col] = (vals - mn) / (mx - mn)
    return out

# ============================================================
# Figure 1: Overall attribution quality
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 5))

data, labels, colors = prepare_metric_groups(summary_df, "cosine_truegrad_mean")
colored_boxplot(axes[0], data, labels, colors, "Cosine Similarity", "(a) Cosine to True Gradient")

data, labels, colors = prepare_metric_groups(summary_df, "boundary_alignment_mean")
colored_boxplot(axes[1], data, labels, colors, "Boundary Alignment", "(b) Boundary Alignment")
axes[1].axhline(0, linewidth=1, color="black")

fig.suptitle("Overall Attribution Quality Across Datasets", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTDIR / "5_3_fig1_overall_quality_boxplots.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# Figure 2: Fidelity
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 5))

data, labels, colors = prepare_metric_groups(summary_df, "fidelity_del_auc_mean")
colored_boxplot(axes[0], data, labels, colors, "Deletion AUC", "(a) Deletion AUC")

data, labels, colors = prepare_metric_groups(summary_df, "fidelity_ins_auc_mean")
colored_boxplot(axes[1], data, labels, colors, "Insertion AUC", "(b) Insertion AUC")

fig.suptitle("Fidelity Across Datasets", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTDIR / "5_3_fig2_fidelity_boxplots.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# Figure 3: Stability
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 5))

data, labels, colors = prepare_metric_groups(summary_df, "stability_lipschitz_mean")
colored_boxplot(axes[0], data, labels, colors, "Lipschitz Stability", "(a) Lipschitz Stability")

data, labels, colors = prepare_metric_groups(summary_df, "stability_mean_attr_change_mean")
colored_boxplot(axes[1], data, labels, colors, "Mean Attribution Change", "(b) Mean Attribution Change")

fig.suptitle("Stability Across Datasets", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTDIR / "5_3_fig3_stability_boxplots.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# Figure 4: Dataset-level heatmap
# ============================================================
heat_metric = "cosine_truegrad_mean"
heat_title = "Cosine to True Gradient"

heat_df = summary_df.pivot(index="dataset", columns="group", values=heat_metric)
heat_df = heat_df.reindex(index=dataset_order, columns=group_order)
heat_df.index = [dataset_label_map.get(idx, idx) for idx in heat_df.index]
heat_df.columns = [group_label_map.get(col, col) for col in heat_df.columns]

norm_heat_df = columnwise_normalize(heat_df)

fig, ax = plt.subplots(figsize=(12, 5.5))
im = ax.imshow(norm_heat_df.values, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")

ax.set_xticks(np.arange(len(norm_heat_df.columns)))
ax.set_yticks(np.arange(len(norm_heat_df.index)))
ax.set_xticklabels(norm_heat_df.columns, rotation=35, ha="right", fontsize=10)
ax.set_yticklabels(norm_heat_df.index, fontsize=10)

for i in range(norm_heat_df.shape[0]):
    for j in range(norm_heat_df.shape[1]):
        original_val = heat_df.iloc[i, j]
        if pd.isna(original_val):
            text = "NA"
            val_for_color = 0.0
        else:
            text = f"{original_val:.2f}"
            val_for_color = norm_heat_df.iloc[i, j]
        ax.text(
            j, i, text,
            ha="center", va="center",
            fontsize=8,
            color="black" if val_for_color < 0.6 else "white"
        )

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Column-wise normalized value", fontsize=10)

ax.set_title(f"Dataset-Level Overview: {heat_title}", fontsize=13)
plt.tight_layout()
plt.savefig(OUTDIR / "5_3_fig4_dataset_heatmap_cosine.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# Prepare disagreement data
# Supports either:
#   method_a/method_b already like ig_zero, ig_boundarycrawler, ig_gs
# or:
#   method_a/baseline_a/method_b/baseline_b
# ============================================================
disagreement_df["dataset"] = pd.Categorical(disagreement_df["dataset"], categories=dataset_order, ordered=True)
disagreement_df = disagreement_df.sort_values("dataset").copy()

if {"baseline_a", "baseline_b"}.issubset(disagreement_df.columns):
    disagreement_df["method_a"] = disagreement_df["method_a"].map(method_map).fillna(disagreement_df["method_a"])
    disagreement_df["baseline_a"] = disagreement_df["baseline_a"].map(baseline_map).fillna(disagreement_df["baseline_a"])
    disagreement_df["method_b"] = disagreement_df["method_b"].map(method_map).fillna(disagreement_df["method_b"])
    disagreement_df["baseline_b"] = disagreement_df["baseline_b"].map(baseline_map).fillna(disagreement_df["baseline_b"])

    disagreement_df["group_a"] = disagreement_df["method_a"] + "_" + disagreement_df["baseline_a"]
    disagreement_df["group_b"] = disagreement_df["method_b"] + "_" + disagreement_df["baseline_b"]
else:
    group_name_map = {
        # old names
        "ig_zero": "IG_Zero",
        "ig_boundary": "IG_BC",
        "lime_zero": "LIME_Zero",
        "lime_boundary": "LIME_BC",
        "ks_zero": "KS_Zero",
        "ks_boundary": "KS_BC",

        # new names
        "ig_boundarycrawler": "IG_BC",
        "ig_gs": "IG_GS",
        "lime_boundarycrawler": "LIME_BC",
        "lime_gs": "LIME_GS",
        "ks_boundarycrawler": "KS_BC",
        "ks_gs": "KS_GS",

        # short aliases
        "ig_bc": "IG_BC",
        "lime_bc": "LIME_BC",
        "ks_bc": "KS_BC",

        # already formatted names
        "IG_Zero": "IG_Zero",
        "IG_BC": "IG_BC",
        "IG_GS": "IG_GS",
        "LIME_Zero": "LIME_Zero",
        "LIME_BC": "LIME_BC",
        "LIME_GS": "LIME_GS",
        "KS_Zero": "KS_Zero",
        "KS_BC": "KS_BC",
        "KS_GS": "KS_GS",
    }
    disagreement_df["group_a"] = disagreement_df["method_a"].replace(group_name_map)
    disagreement_df["group_b"] = disagreement_df["method_b"].replace(group_name_map)

# keep only supported groups
disagreement_df = disagreement_df[
    disagreement_df["group_a"].isin(group_order) & disagreement_df["group_b"].isin(group_order)
].copy()

# ============================================================
# Figure 5: Pairwise disagreement heatmaps
# ============================================================
pretty_method_labels = [group_label_map[g] for g in group_order]

def build_sym_matrix(df_in: pd.DataFrame, metric: str):
    mat = pd.DataFrame(np.nan, index=group_order, columns=group_order)

    for g in group_order:
        mat.loc[g, g] = 1.0 if metric in {"jaccard_topk", "spearman_rank"} else 0.0

    grouped = (
        df_in.groupby(["group_a", "group_b"], dropna=False)[metric]
        .mean()
        .reset_index()
    )

    for _, row in grouped.iterrows():
        a, b, val = row["group_a"], row["group_b"], row[metric]
        if a in mat.index and b in mat.columns:
            mat.loc[a, b] = val
            mat.loc[b, a] = val

    return mat.reindex(index=group_order, columns=group_order)

metrics = [
    ("jaccard_topk", "Jaccard Top-k"),
    ("spearman_rank", "Spearman Rank"),
    ("delta_l2", r"$\Delta L_2$"),
]

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for ax, (metric, title) in zip(axes, metrics):
    mat = build_sym_matrix(disagreement_df, metric)

    vals = mat.values.astype(float)
    if np.all(np.isnan(vals)):
        vmin, vmax = 0.0, 1.0
    else:
        vmin = np.nanmin(vals)
        vmax = np.nanmax(vals)
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-8

    im = ax.imshow(vals, aspect="equal", vmin=vmin, vmax=vmax, cmap="coolwarm")

    ax.set_xticks(np.arange(len(pretty_method_labels)))
    ax.set_yticks(np.arange(len(pretty_method_labels)))
    ax.set_xticklabels(pretty_method_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(pretty_method_labels, fontsize=9)
    ax.set_title(title, fontsize=12)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            text = "NA" if pd.isna(val) else f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color="black")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

fig.suptitle("Pairwise Disagreement Between Explanation Settings", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(OUTDIR / "5_3_fig5_pairwise_disagreement_heatmaps.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved plots to: {OUTDIR}")