from utils import (
    generate_linearly_separable_data,
    save_dataset_to_csv,
    plot_2d_dataset,
)

X, y = generate_linearly_separable_data(
    n_samples_per_class=200,
    n_features=2,
    random_state=42,
)

csv_path = save_dataset_to_csv(X, y, out_dir="../data/synthetic")
plot_path = plot_2d_dataset(X, y, out_dir="../figures", show=False)

print("CSV saved at:", csv_path)
print("Plot saved at:", plot_path)
