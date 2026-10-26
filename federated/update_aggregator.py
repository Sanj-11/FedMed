from collections.abc import Sequence
import io

import numpy as np
import torch

from federated.fedavg import aggregate_parameters


def _decode_model_update_with_names(
    payload: bytes,
) -> tuple[list[str], list[np.ndarray]]:
    """Decode a serialized PyTorch state dictionary, preserving parameter names."""
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("payload must be non-empty bytes")

    try:
        state_dict = torch.load(
            io.BytesIO(payload),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise ValueError("Invalid serialized model update") from exc

    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Model update must contain a non-empty state dictionary")

    names = []
    arrays = []

    for key, tensor in state_dict.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("Model update must contain named tensors")

        array = tensor.detach().cpu().numpy().copy()

        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"Model update tensor {key!r} must be numeric")

        if not np.isfinite(array).all():
            raise ValueError(
                f"Model update tensor {key!r} contains non-finite values"
            )

        names.append(key)
        arrays.append(array)

    return names, arrays


def decode_model_update(payload: bytes) -> list[np.ndarray]:
    """Decode a serialized PyTorch state dictionary into NumPy arrays."""
    _, arrays = _decode_model_update_with_names(payload)
    return arrays


def aggregate_serialized_updates(
    updates: Sequence[bytes],
    num_examples: Sequence[int],
) -> list[np.ndarray]:
    """Decode serialized updates and aggregate them using weighted FedAvg."""
    if not updates:
        raise ValueError("updates must not be empty")

    if len(updates) != len(num_examples):
        raise ValueError("Each update must have a corresponding example count")

    decoded = [
        _decode_model_update_with_names(payload)
        for payload in updates
    ]

    expected_names = decoded[0][0]

    for names, _ in decoded[1:]:
        if names != expected_names:
            raise ValueError(
                "All updates must have matching parameter names and ordering"
            )

    arrays_by_client = [arrays for _, arrays in decoded]

    return aggregate_parameters(arrays_by_client, num_examples)