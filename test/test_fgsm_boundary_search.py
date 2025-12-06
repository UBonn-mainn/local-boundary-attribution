# tests/test_fgsm_boundary_search.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from boundary_search.fgsm import fgsm_boundary_search


class SimpleLinearBinary(nn.Module):
    """
    A simple binary classifier in 2D.
    Decision boundary is roughly a line in R^2.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            # Make class 0 prefer negative x, class 1 prefer positive x
            self.linear.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
            self.linear.bias.zero_()

    def forward(self, x):
        # x: (batch, 2)
        return self.linear(x)


class ConstantModel(nn.Module):
    """
    Always predicts the same class, no matter the input.
    Useful to test failure mode where no flip can occur.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Two logits, always favor class 0
        batch = x.shape[0]
        logits = torch.zeros(batch, 2)
        logits[:, 0] = 1.0
        return logits


def test_boundary_search_returns_tensor_and_flag():
    model = SimpleLinearBinary()
    x = torch.tensor([2.0, 0.0])  # clearly on one side of boundary (class 0)

    boundary_x, success = fgsm_boundary_search(model, x, step_size=0.5, max_iters=10)

    assert isinstance(boundary_x, torch.Tensor)
    assert isinstance(success, bool)
    # shape should be same as original
    assert boundary_x.shape == x.shape


def test_boundary_search_changes_class_somewhere():
    """
    Sanity check: if success=True, we expect that we crossed the boundary,
    i.e., moving a tiny bit from boundary in either direction flips class.
    """
    model = SimpleLinearBinary()
    x = torch.tensor([2.0, 0.0])  # start clearly in class 0

    boundary_x, success = fgsm_boundary_search(model, x, step_size=0.5, max_iters=20)

    assert success, "Boundary search should succeed for this simple model."

    with torch.no_grad():
        # Original class
        orig_logits = model(x.unsqueeze(0))
        orig_class = orig_logits.argmax(dim=1)[0].item()

        # Check classification slightly to one side and the other
        eps = 1e-2
        # Move a bit in negative x direction (towards class 1 region)
        x_minus = boundary_x.clone()
        x_minus[0] -= eps
        cls_minus = model(x_minus.unsqueeze(0)).argmax(dim=1)[0].item()

        # Move a bit in positive x direction (towards class 0 region)
        x_plus = boundary_x.clone()
        x_plus[0] += eps
        cls_plus = model(x_plus.unsqueeze(0)).argmax(dim=1)[0].item()

    # At least one side should differ from original class
    assert (cls_minus == orig_class) or (cls_plus == orig_class)
    assert cls_minus != cls_plus, "Classes on the two sides of boundary should differ."


def test_clamp_is_respected():
    model = SimpleLinearBinary()
    x = torch.tensor([2.0, 0.0])

    clamp_range = (-0.5, 0.5)
    boundary_x, success = fgsm_boundary_search(
        model,
        x,
        step_size=1.0,
        max_iters=20,
        clamp=clamp_range,
    )

    low, high = clamp_range
    assert torch.all(boundary_x >= low - 1e-6)
    assert torch.all(boundary_x <= high + 1e-6)


def test_failure_when_no_flip_possible():
    """
    If the model always predicts the same class, FGSM should never flip
    and the function must return success=False.
    """
    model = ConstantModel()
    x = torch.tensor([0.1, 0.2])

    boundary_x, success = fgsm_boundary_search(
        model,
        x,
        step_size=0.1,
        max_iters=10,
        clamp=None,
    )

    assert success is False
    # Should still return a tensor with the same shape
    assert boundary_x.shape == x.shape


def test_raises_on_wrong_shape():
    model = SimpleLinearBinary()
    x = torch.ones(1, 1, 1, 1, 1)  # 5D tensor -> unsupported

    import pytest
    with pytest.raises(ValueError):
        fgsm_boundary_search(model, x)
