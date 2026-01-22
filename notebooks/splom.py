import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
import scipy


# Load dataset
df = pd.read_csv("/Users/nguyennhatmai/Download/evaluation_metrics.csv")

numeric_df = df.select_dtypes(include=[np.number])

# (Optional) focus on key FGSM vs GS variables
cols_of_interest = [
    "fgsm_steps",
    "gs_radius_found",
    "dist_x_to_fgsm_boundary",
    "dist_x_to_gs_boundary",
    "dist_fgsm_boundary_to_gs_boundary",
    "angle_x_fgsm_gs_deg",
    "curve_circle_area",
    "ring_n",
    "ring_r_min",
    "ring_r_max",
    "ring_count_y0",
    "ring_count_y1"
]

numeric_df = numeric_df[cols_of_interest]

# Plot SPLOM
axes = scatter_matrix(
    numeric_df,
    figsize=(12, 12),
    diagonal="kde",   # or "hist"
    alpha=0.6
)

# Rotate axis labels 45 degrees
for ax in axes.flatten():
    if ax is not None:
        ax.xaxis.label.set_rotation(45)
        ax.yaxis.label.set_rotation(0)
        ax.tick_params(axis="x", labelrotation=45)
        ax.tick_params(axis="y", labelrotation=0)

plt.suptitle("Scatter Plot Matrix (FGSM vs GS)", y=1.02)
plt.tight_layout()
plt.show()