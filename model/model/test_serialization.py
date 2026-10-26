import torch

from model.network import SmallSegmentationModel
from model.model.serialization import (
    load_model_parameters,
    serialize_model_parameters,
)


def test_model_parameters_round_trip():
    torch.manual_seed(42)

    original_model = SmallSegmentationModel()
    original_model.eval()

    # Serialize the original model's parameters.
    payload = serialize_model_parameters(original_model)

    assert isinstance(payload, bytes)
    assert len(payload) > 0

    # Create a fresh model and restore the parameters.
    restored_model = SmallSegmentationModel()
    load_model_parameters(restored_model, payload)
    restored_model.eval()

    # Confirm that both models produce the same output.
    sample_input = torch.randn(1, 1, 128, 128)

    with torch.no_grad():
        original_output = original_model(sample_input)
        restored_output = restored_model(sample_input)

    assert torch.allclose(original_output, restored_output)


def test_empty_payload_is_rejected():
    model = SmallSegmentationModel()

    try:
        load_model_parameters(model, b"")
    except ValueError as error:
        assert "non-empty bytes" in str(error)
    else:
        raise AssertionError("Expected ValueError for an empty payload")
