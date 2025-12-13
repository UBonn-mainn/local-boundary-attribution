import torch
import torch.nn as nn
from typing import Tuple

class SimpleClassifier(nn.Module):
    """
    A simple MLP classifier for the tabular datasets.
    Architecture: Input -> Linear(64) -> ReLU -> Linear(64) -> ReLU -> Linear(2)
    """
    def __init__(self, input_dim: int = 2, num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


class LinearClassifier(nn.Module):
    """
    A strictly linear classifier (Logistic Regression).
    Architecture: Input -> Linear(2)
    """
    def __init__(self, input_dim: int = 2, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def load_model(model_path: str, input_dim: int = 2, model_type: str = "mlp") -> nn.Module:
    """
    Load a trained classifier from disk.
    
    Args:
        model_path: Path to the .pth checkpoint.
        input_dim: Input dimension of the data (default 2).
        model_type: "mlp" or "linear".
        
    Returns:
        The loaded model in eval mode.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == "mlp":
        model = SimpleClassifier(input_dim=input_dim)
    elif model_type == "linear":
        model = LinearClassifier(input_dim=input_dim)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    return model
