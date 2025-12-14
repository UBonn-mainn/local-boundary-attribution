import torch
import torch.nn as nn
import torch.nn.functional as F

from boundary_search.fgsm import fgsm_boundary_search


class SimpleLinearBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
            self.linear.bias.zero_()

    def forward(self, x):
        return self.linear(x)


def test_fgsm_finds_boundary_with_balanced_probs():
    model = SimpleLinearBinary()
    x = torch.tensor([-3.0, -3.0])

    boundary_x, success = fgsm_boundary_search(
        model,
        x,
        step_size=0.1,
        max_iters=100,
        clamp=None,
        refine_steps=10,
    )

    assert success, "FGSM search should succeed for the simple linear model."

    with torch.no_grad():
        logits = model(boundary_x.unsqueeze(0))
        probs = F.softmax(logits, dim=1)[0]

    # Boundary should be near equal probability for both classes
    assert torch.allclose(probs[0], torch.tensor(0.5), atol=0.15)
    assert torch.allclose(probs[1], torch.tensor(0.5), atol=0.15)
    # Point should be close to the decision boundary x=0
    assert abs(boundary_x[0].item()) < 0.25