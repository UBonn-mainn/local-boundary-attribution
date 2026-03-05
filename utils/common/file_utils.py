import re
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

def collect_synthetic_experiments_map(
        root="/Users/nguyennhatmai/Documents/study/UBonn/WiSe2526/LabDMAI/local-boundary-attribution/results/synthetic_experiments",
):
    """
    Traverse direct child folders of `root` and return a dictionary:

    {
        folder_name: {
            "path": str,
            "dimension": int,
            "dataset_type": str,
            "n_classes": int
        }
    }
    """

    root_path = Path(root).expanduser().resolve()

    if not root_path.exists():
        raise FileNotFoundError(f"Path does not exist: {root_path}")

    result = {}

    dim_pattern = re.compile(r"^(\d+)d$", re.IGNORECASE)
    class_pattern = re.compile(r"^(\d+)class$", re.IGNORECASE)

    for child in sorted(root_path.iterdir()):
        if not child.is_dir():
            continue

        folder_name = child.name
        parts = folder_name.split("_")

        # ---- dimension ----
        dim_match = dim_pattern.match(parts[0])
        if not dim_match:
            continue  # skip unexpected folders

        dimension = int(dim_match.group(1))

        # ---- dataset type ----
        dataset_type = "_".join(parts[1:]) if len(parts) > 1 else ""

        # ---- number of classes ----
        n_classes = 2  # default
        for token in parts[1:]:
            class_match = class_pattern.match(token)
            if class_match:
                n_classes = int(class_match.group(1))
                break

        result[folder_name] = {
            "path": str(child),
            "dimension": dimension,
            "dataset_type": dataset_type,
            "n_classes": n_classes,
        }

    return result

