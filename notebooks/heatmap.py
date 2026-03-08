import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
dataset = '8d_linear_blobs'
# dataset = '15d_moons'
# dataset = '15d_3class'
# dataset = '8d_10class'
df = pd.read_csv(f"/Users/nguyennhatmai/Documents/study/UBonn/WiSe2526/LabDMAI/local-boundary-attribution/results/synthetic_experiments/{dataset}/report.csv")

numeric_df = df.select_dtypes(include=[np.number])
numeric_df["radii"] = abs(numeric_df["ring_r_max"] - numeric_df["ring_r_min"])


# (Optional) focus on key FGSM vs GS variables
cols_of_interest = [
    # "fgsm_steps",
    # "gs_radius_found",
    "dist_x_to_crawler",
    "dist_x_to_gs",
    "dist_crawler_to_gs",
    "sphere_vol",
    # "ring_n",
    # "ring_r_min",
    # "ring_r_max",
    # "ring_count_y0",
    # "ring_count_y1",
    "radii",
    "angle_x_crawler_gs_deg",
]

# plt.scatter(numeric_df["angle_x_crawler_gs_deg"], numeric_df["sphere_vol"])
# plt.show()

numeric_df = numeric_df[cols_of_interest]

# Compute correlation matrix (Pearson by default)
corr = numeric_df.corr()
rename_map = {
    "dist_x_to_crawler": r"$d_{\mathrm{BC}}$",
    "dist_x_to_gs": r"$d_{\mathrm{GS}}$",
    "dist_crawler_to_gs": r"$d_{between}$",
    "sphere_vol": r"Vol",
    "radii": r"radii",
    "angle_x_crawler_gs_deg": r"$\theta$"
}

corr_plot = corr.rename(index=rename_map, columns=rename_map)
corr_plot = corr_plot.fillna(0)

# # Plot correlation heatmap
# plt.figure(figsize=(10, 8))
# plt.imshow(corr, interpolation="nearest")
# plt.colorbar(label="Correlation coefficient")
#
# plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
# plt.yticks(range(len(corr.columns)), corr.columns)
#
# plt.title("Correlation Matrix")
# plt.tight_layout()
# plt.show()


# Plot with diverging colormap
# plt.figure(figsize=(10, 8))
# im = plt.imshow(
#     corr,
#     cmap="coolwarm",   # diverging colormap
#     vmin=-1, vmax=1    # symmetric range around 0
# )
#
# plt.colorbar(im, label="Correlation coefficient")
#
# plt.xticks(
#     range(len(corr.columns)),
#     corr.columns,
#     rotation=45,
#     ha="right"
# )
# plt.yticks(
#     range(len(corr.columns)),
#     corr.columns
# )
#
# plt.title("Correlation Matrix (Diverging Colormap)")
# plt.tight_layout()
# plt.show()


# Plot heatmap
fig, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(
    corr,
    cmap="coolwarm",   # diverging colormap
    vmin=-1,
    vmax=1
)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Correlation coefficient")

# Axis labels
ax.set_xticks(range(len(corr_plot.columns)))
ax.set_yticks(range(len(corr_plot.columns)))
ax.set_xticklabels(corr_plot.columns, fontsize=20)
ax.set_yticklabels(corr_plot.columns, fontsize=20)

# Annotate each cell with the correlation value
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        value = corr_plot.iloc[i, j]
        ax.text(
            j, i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            color="black" if abs(value) < 0.6 else "white"
        )

ax.set_title("Correlation Matrix with Values")
plt.tight_layout()
plt.savefig(f"./heatmap/{dataset}_heatmap.png")
# plt.show()