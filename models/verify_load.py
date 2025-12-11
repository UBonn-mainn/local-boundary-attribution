
import torch
import numpy as np
from models.load_model import load_model

def test_loading():
    model_path = "models/checkpoints/test_model.pth"
    print(f"Loading model from {model_path}...")
    
    try:
        model = load_model(model_path, input_dim=2)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Create dummy input based on the range of synthetic data (-5 to 7 roughly)
    dummy_input = torch.tensor([[ -2.0, 0.0], [5.0, 5.0]], dtype=torch.float32)
    
    print("Running inference...")
    with torch.no_grad():
        logits = model(dummy_input)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
    print("Logits:\n", logits)
    print("Probs:\n", probs)
    print("Predictions:\n", preds)
    
    # Simple sanity check
    # [-2, 0] likely Class 0 (mean neg)
    # [5, 5] likely Class 1 (mean pos)
    assert preds[0].item() == 0, "Expected Class 0 for negative input"
    assert preds[1].item() == 1, "Expected Class 1 for positive input"
    print("Sanity check passed!")

if __name__ == "__main__":
    test_loading()
