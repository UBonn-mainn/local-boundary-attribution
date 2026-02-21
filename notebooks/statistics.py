import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon, mannwhitneyu

ROOT = Path("/Users/nguyennhatmai/Documents/study/UBonn/WiSe2526/LabDMAI/local-boundary-attribution/results/synthetic_experiments")
OUT_CSV = "synthetic_experiments_summary.csv"

# -------------------------
# Helpers
# -------------------------
def safe_mean(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.mean()) if len(s) else np.nan

def safe_median(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.median()) if len(s) else np.nan

def safe_std(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.std()) if len(s) else np.nan

def safe_iqr(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.quantile(0.75) - s.quantile(0.25)) if len(s) else np.nan

def wilcoxon_with_effect(a, b):
    try:
        stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        n = len(a)
        rbc = 1 - (2 * stat) / (n * (n + 1))
        return float(p), float(rbc)
    except Exception:
        return np.nan, np.nan

def mannwhitney_safe(a, b):
    try:
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        return float(p)
    except Exception:
        return np.nan

def coerce_bool_col(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        return
    if df[col].dtype == bool:
        return
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(df[col])
    )
    if df[col].dtype != bool:
        df[col] = df[col].astype(int).astype(bool)

# -------------------------
# Metrics to summarize
# -------------------------
METRICS = [
    "angle_x_crawler_gs_deg",
    "sphere_vol",
    "ring_n",
    "dist_x_to_crawler",
    "dist_x_to_gs",
    "dist_crawler_to_gs",
]

results = []

for report_path in ROOT.rglob("report.csv"):
    dataset = report_path.parent.name
    df = pd.read_csv(report_path)

    coerce_bool_col(df, "crawler_success")
    coerce_bool_col(df, "gs_success")

    if "crawler_success" not in df.columns or "gs_success" not in df.columns:
        continue

    # subset where BOTH are successful
    df_both = df[df["crawler_success"] & df["gs_success"]]

    row = {
        "dataset": dataset,
        "report_path": str(report_path),
        "n_samples": int(len(df)),
        "n_both_success": int(len(df_both)),
        "crawler_success_rate": float(df["crawler_success"].mean()),
        "gs_success_rate": float(df["gs_success"].mean()),
        "both_success_rate": float((df["crawler_success"] & df["gs_success"]).mean()),
        "agree_success_rate": float((df["crawler_success"] == df["gs_success"]).mean()),
        "crawler_only_success_rate": float((df["crawler_success"] & ~df["gs_success"]).mean()),
        "gs_only_success_rate": float((~df["crawler_success"] & df["gs_success"]).mean()),
    }

    # -------------------------
    # Metric summaries ONLY on df_both
    # -------------------------
    for m in METRICS:
        if m in df.columns:
            row[f"{m}_mean"] = safe_mean(df_both[m])
            row[f"{m}_median"] = safe_median(df_both[m])
            row[f"{m}_std"] = safe_std(df_both[m])
            row[f"{m}_iqr"] = safe_iqr(df_both[m])
        else:
            row[f"{m}_mean"] = np.nan
            row[f"{m}_median"] = np.nan
            row[f"{m}_std"] = np.nan
            row[f"{m}_iqr"] = np.nan

    # -------------------------
    # Paired tests ONLY on df_both as well
    # -------------------------
    req = {"dist_x_to_crawler", "dist_x_to_gs"}
    if req.issubset(df.columns):
        df_pair = df_both.dropna(subset=list(req))
        row["n_paired"] = int(len(df_pair))

        if len(df_pair) >= 5:
            c = df_pair["dist_x_to_crawler"].to_numpy()
            g = df_pair["dist_x_to_gs"].to_numpy()

            diff = c - g
            row["dist_diff_mean"] = float(np.mean(diff))
            row["dist_diff_median"] = float(np.median(diff))

            row["wilcoxon_p"], row["wilcoxon_rbc"] = wilcoxon_with_effect(c, g)
            row["mannwhitney_p"] = mannwhitney_safe(c, g)
        else:
            row["dist_diff_mean"] = np.nan
            row["dist_diff_median"] = np.nan
            row["wilcoxon_p"] = np.nan
            row["wilcoxon_rbc"] = np.nan
            row["mannwhitney_p"] = np.nan
    else:
        row["n_paired"] = np.nan
        row["dist_diff_mean"] = np.nan
        row["dist_diff_median"] = np.nan
        row["wilcoxon_p"] = np.nan
        row["wilcoxon_rbc"] = np.nan
        row["mannwhitney_p"] = np.nan

    results.append(row)

summary_df = (
    pd.DataFrame(results)
    .sort_values("dataset")
    .reset_index(drop=True)
)

print(summary_df)
summary_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")
