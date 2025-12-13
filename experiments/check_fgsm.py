
import torch
from utils.data.load_model import load_model
from boundary_search.fgsm import fgsm_boundary_search

def check_fgsm():
    # Load the linear model
    model_path = "models/checkpoints/linear_model.pth"
    print(f"Loading linear model from {model_path}...")
    try:
        model = load_model(model_path, input_dim=2, model_type="linear")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Create a dummy input far from the boundary (e.g., class 0 area)
    # Based on previous data gen, class 0 is around (-2, -2) roughly
    x_input = torch.tensor([-3.0, -3.0], dtype=torch.float32)
    
    print(f"Original point: {x_input}")
    
    # Run FGSM search
    # We use a larger step size to ensure it reaches the boundary quickly for this test
    # clamp=None because our data isn't image data bounded in [0,1]
    boundary_point, success = fgsm_boundary_search(
        model, 
        x_input, 
        step_size=0.1, 
        max_iters=100, 
        clamp=None, 
        refine_steps=10
    )
    
    print(f"Search Success: {success}")
    print(f"Boundary point: {boundary_point}")
    
    if success:
        with torch.no_grad():
            logits = model(boundary_point.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)
            print(f"Logits at boundary: {logits}")
            print(f"Probs at boundary: {probs}")
            
            # Ideally probs should be close to [0.5, 0.5]
            if torch.abs(probs[0, 0] - 0.5) < 0.1:
                print("VERIFIED: Point is near decision boundary (probs ~ 0.5).")
            else:
                print("WARNING: Point might not be exactly on boundary.")

if __name__ == "__main__":
    check_fgsm()
