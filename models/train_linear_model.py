
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path
from models.load_model import LinearClassifier
from utils.dataset_utils import load_dataset_from_csv

def train(args):
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 1. Load Data
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")
        
    print(f"Loading data from {data_path}...")
    X_np, y_np = load_dataset_from_csv(data_path)
    X_np = X_np.astype(np.float32)
    y_np = y_np.astype(np.int64)
    
    input_dim = X_np.shape[1]
    
    # Convert to Tensor
    dataset = TensorDataset(torch.from_numpy(X_np), torch.from_numpy(y_np))
    
    # Split train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2. Init Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = LinearClassifier(input_dim=input_dim, num_classes=2)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 3. Training Loop
    best_val_acc = 0.0
    
    save_dir = Path(args.save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting training (Linear Model)...")
    for epoch in range(args.epochs):
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
            
        train_loss /= len(train_dataset)
        
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
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
        # Save best model
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            
    print(f"Training complete. Best Val Acc: {best_val_acc:.4f}")
    print(f"Model saved to {args.save_path}")


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
