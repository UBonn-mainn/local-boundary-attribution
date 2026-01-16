import argparse
from pathlib import Path
from utils.data import generate_concentric_hyperspheres_data, save_dataset_to_csv, plot_2d_dataset

def main():
    parser = argparse.ArgumentParser(description="Generate Concentric Hyperspheres Dataset")
    parser.add_argument("--n_samples", type=int, default=500, help="Samples per class")
    parser.add_argument("--dims", type=int, default=8, help="Number of dimensions")
    parser.add_argument("--out_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--noise", type=float, default=0.1, help="Noise level")
    
    args = parser.parse_args()
    
    print(f"Generating {args.dims}D Concentric Hyperspheres Data...")
    X, y = generate_concentric_hyperspheres_data(
        n_samples_per_class=args.n_samples,
        n_features=args.dims,
        noise=args.noise
    )
    
    prefix = f"hyperspheres_{args.dims}d"
    csv_path = save_dataset_to_csv(X, y, out_dir=args.out_dir, prefix=prefix)
    print(f"Data saved to: {csv_path}")
    
    # Optional: Plot first 2 dims
    if args.dims == 2:
        plot_file = plot_2d_dataset(X, y, out_dir=args.out_dir, prefix=prefix + "_plot", show=False)
        print(f"2D Projection saved to: {plot_file}")

if __name__ == "__main__":
    main()
