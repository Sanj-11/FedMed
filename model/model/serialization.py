
import io
from collections.abc import Mapping

import torch
from torch import nn


def serialize_model_parameters(model: nn.Module) -> bytes:
    """Serialize a PyTorch model's state dictionary into bytes."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def load_model_parameters(model: nn.Module, payload: bytes) -> None:
    """Load serialized model parameters into an existing model."""
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("payload must be non-empty bytes")

    buffer = io.BytesIO(payload)
    state_dict = torch.load(
        buffer,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(state_dict, Mapping):
        raise ValueError("payload must contain a model state dictionary")

    # Strict loading checks that parameter names and shapes match.
    model.load_state_dict(state_dict, strict=True)
