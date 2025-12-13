import torch
import pytest

from utils.data import LinearClassifier, SimpleClassifier, load_model


@pytest.mark.parametrize(
    "model_type, model_cls",
    [
        ("mlp", SimpleClassifier),
        ("linear", LinearClassifier),
    ],
)
def test_load_model_round_trip(tmp_path, model_type, model_cls):
    torch.manual_seed(0)
    source_model = model_cls(input_dim=2)

    checkpoint_path = tmp_path / f"{model_type}_checkpoint.pth"
    torch.save(source_model.state_dict(), checkpoint_path)

    loaded_model = load_model(str(checkpoint_path), input_dim=2, model_type=model_type)

    device = next(loaded_model.parameters()).device
    source_model.to(device)
    source_model.eval()

    dummy_input = torch.tensor([[-2.0, 0.0], [5.0, 5.0]], dtype=torch.float32, device=device)

    with torch.no_grad():
        expected_logits = source_model(dummy_input)
        loaded_logits = loaded_model(dummy_input)

    torch.testing.assert_close(loaded_logits, expected_logits)
    assert not loaded_model.training