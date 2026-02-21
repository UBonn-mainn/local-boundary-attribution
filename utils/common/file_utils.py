from pathlib import Path

import numpy as np


def find_data_csv_folders(root_dir: str):
    """
    Recursively search root_dir and return all folders
    that contain a file named 'data.csv'.
    """
    root = Path(root_dir)

    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    folders = []

    # Recursively search for data.csv
    for data_file in root.rglob("data.csv"):
        folders.append(str(data_file.parent))
        # print(data_file.parent)

    return sorted(folders)


def to_row(
        *,
        x0: np.ndarray,
        y_true: int,
        gs_res,
):
    d = x0.shape[0]
    row = {
        "num_feature": int(d)
    }

    # original point
    for j in range(d):
        row[f"feature_{j + 1}"] = float(x0[j])
    row["class"] = int(y_true)
    row["gs_success"] = bool(gs_res.success)
    row["radius_found"] = float(gs_res.radius_found)

    # enemy + boundary
    if gs_res.success:
        for j in range(d):
            row[f"x_enemy_{j}"] = float(gs_res.x_enemy[j])
        for j in range(d):
            row[f"x_boundary_{j}"] = float(gs_res.x_boundary[j])

        # distances (useful diagnostics)
        row["dist_x0_enemy"] = float(np.linalg.norm(gs_res.x_enemy - x0))
        row["dist_x0_boundary"] = float(np.linalg.norm(gs_res.x_boundary - x0))
    else:
        for j in range(d):
            row[f"x_enemy_{j}"] = np.nan
        for j in range(d):
            row[f"x_boundary_{j}"] = np.nan
        row["dist_x0_enemy"] = np.nan
        row["dist_x0_boundary"] = np.nan

    return row
