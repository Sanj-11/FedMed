from collections import OrderedDict

import numpy as np
import torch
from torch import nn


def get_model_parameters(model: nn.Module) -> list[np.ndarray]:
    """Return model parameters as independent NumPy arrays."""
    return [
        tensor.detach().cpu().numpy().copy()
        for tensor in model.state_dict().values()
    ]


def set_model_parameters(
    model: nn.Module,
    parameters: list[np.ndarray],
) -> None:
    """Load NumPy parameter arrays into a model."""
    current_state = model.state_dict()
    expected_keys = list(current_state.keys())

    if len(parameters) != len(expected_keys):
        raise ValueError(
            f"Expected {len(expected_keys)} parameter arrays, "
            f"received {len(parameters)}"
        )

    updated_state = OrderedDict()

    for key, array in zip(expected_keys, parameters):
        expected_tensor = current_state[key]

        if tuple(array.shape) != tuple(expected_tensor.shape):
            raise ValueError(
                f"Shape mismatch for {key}: expected "
                f"{tuple(expected_tensor.shape)}, received {tuple(array.shape)}"
            )

        updated_state[key] = torch.as_tensor(
            array,
            dtype=expected_tensor.dtype,
            device=expected_tensor.device,
        )

    model.load_state_dict(updated_state, strict=True)