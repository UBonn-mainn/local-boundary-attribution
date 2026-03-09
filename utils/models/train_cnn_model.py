import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from utils.data.dataset_utils import get_mnist_dataloaders
from utils.entities.small_mnist_cnn import SmallMNISTCNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def train(model: nn.Module, train_loader, test_loader, device: torch.device, epochs: int, lr: float) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)

        train_acc = correct / max(total, 1)
        test_acc = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"loss={running_loss / max(total, 1):.4f} | "
            f"train_acc={train_acc:.4f} | test_acc={test_acc:.4f}"
        )



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/Users/nguyennhatmai/Documents/study/UBonn/WiSe2526/LabDMAI/local-boundary-attribution/data")
    parser.add_argument("--output_dir", type=str, default="/Users/nguyennhatmai/Documents/study/UBonn/WiSe2526/LabDMAI/local-boundary-attribution/models/checkpoints/mnist")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--digit_a", type=int, default=None)
    parser.add_argument("--digit_b", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    binary_digits = None
    if args.digit_a is not None and args.digit_b is not None:
        binary_digits = (args.digit_a, args.digit_b)

    train_loader, test_loader, num_classes = get_mnist_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        binary_digits=binary_digits,
    )

    model = SmallMNISTCNN(num_classes=num_classes).to(device)
    train(model, train_loader, test_loader, device, args.epochs, args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    config = {
        "data_root": args.data_root,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "binary_digits": binary_digits,
        "num_classes": num_classes,
        "model_path": str(model_path),
    }
    with open(output_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()