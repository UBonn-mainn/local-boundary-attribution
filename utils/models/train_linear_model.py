
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path
from utils.data.load_model import LinearClassifier
from utils.data.dataset_utils import load_dataset_from_csv


def train_model_memory(
    X_train: np.ndarray = None, 
    y_train: np.ndarray = None,
    input_dim: int = None,
    num_classes: int = None,
    data_path: str = None,
    val_data: tuple = None,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.01,
    seed: int = 42,
    save_path: str = None
):
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Logic to load data if X_train/y_train not provided
    if X_train is None or y_train is None:
        if data_path is None:
            raise ValueError("Must provide either (X_train, y_train) or 'data_path'.")
        
        path_obj = Path(data_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Data file not found at {data_path}")
            
        print(f"Loading data from {path_obj}...")
        X_train, y_train = load_dataset_from_csv(path_obj)
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.int64)

    if input_dim is None:
        input_dim = X_train.shape[1]
    
    # Auto-detect num_classes from training data
    if num_classes is None:
        num_classes = len(np.unique(y_train))
    
    # Convert to Tensor
    dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    
    # Validation handling
    if val_data:
        X_val, y_val = val_data
        val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    else:
        # Default simple split if no explicit validation data
        val_size = int(len(dataset) * 0.2)
        train_size = len(dataset) - val_size
        dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Init Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = LinearClassifier(input_dim=input_dim, num_classes=num_classes)

    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 3. Training Loop
    best_val_acc = 0.0
    best_model_state = None
    
    print("Starting training (Linear Model)...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
                
        val_acc = correct / total if total > 0 else 0.0
        val_loss /= len(val_dataset)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
        # Save best model state
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            
    print(f"Training complete. Best Val Acc: {best_val_acc:.4f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    if save_path:
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        
    return model

def train(args):
    # 1. Load Data
    train_model_memory(
        data_path=str(args.data_path),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        save_path=args.save_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a linear classifier on CSV data.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--save_path", type=str, default="models/checkpoints/linear_model.pth", help="Path to save the trained model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of data to use for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    train(args)
