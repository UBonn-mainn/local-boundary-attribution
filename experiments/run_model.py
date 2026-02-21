import argparse
from pathlib import Path

from utils.common.file_utils import find_data_csv_folders
from utils.models.train_mlp_model import train_mlp_model_memory

if __name__ == '__main__':
    # root_directory = (
    #     "/Users/nguyennhatmai/Documents/study/UBonn/WiSe2526/LabDMAI/local-boundary-attribution/results"
    # )

    parser = argparse.ArgumentParser()

    # IO
    parser.add_argument("--root_directory", type=str, required=True,
                        default="/Users/nguyennhatmai/Documents/study/UBonn/WiSe2526/LabDMAI/local-boundary-attribution/results")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of data to use for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    root_directory = args.root_directory
    epochs = args.epoch
    batch_size = args.batch_size
    lr = args.lr
    seed = args.seed

    folders = find_data_csv_folders(root_directory)

    for folder in folders:
        print(folder + "/data.csv")
        train_mlp_model_memory(
            data_path=folder + "/data.csv",
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            save_path=folder + "/mlp_model.pth"
        )
