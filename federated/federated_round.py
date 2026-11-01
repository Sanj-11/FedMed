"""Validate and aggregate hospital model updates for a federated round."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def aggregate_hospital_updates(
    updates: Sequence[Mapping[str, Any]],
) -> dict[str, torch.Tensor]:
    """Aggregate state dictionaries using sample-count-weighted FedAvg.

    Each update must contain:
      - state_dict: a mapping of parameter names to tensors
      - num_samples: a positive integer
    """
    if not updates:
        raise ValueError("At least one hospital update is required.")

    reference = updates[0].get("state_dict")
    if not isinstance(reference, Mapping) or not reference:
        raise ValueError("The first update has an invalid state_dict.")

    reference_keys = set(reference)
    validated = []
    total_samples = 0

    for index, update in enumerate(updates):
        if not isinstance(update, Mapping):
            raise ValueError(f"Update {index} must be a mapping.")

        state_dict = update.get("state_dict")
        num_samples = update.get("num_samples")

        if not isinstance(state_dict, Mapping) or not state_dict:
            raise ValueError(f"Update {index} has an invalid state_dict.")

        if set(state_dict) != reference_keys:
            raise ValueError(
                f"Update {index} has incompatible model parameter keys."
            )

        if (
            isinstance(num_samples, bool)
            or not isinstance(num_samples, int)
            or num_samples <= 0
        ):
            raise ValueError(
                f"Update {index} num_samples must be a positive integer."
            )

        for key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"Update {index}, parameter {key!r}, must be a tensor."
                )
            if value.shape != reference[key].shape:
                raise ValueError(
                    f"Update {index}, parameter {key!r}, has an incompatible shape."
                )
            if value.dtype != reference[key].dtype:
                raise ValueError(
                    f"Update {index}, parameter {key!r}, has an incompatible dtype."
                )

        validated.append((state_dict, num_samples))
        total_samples += num_samples

    result = {}

    for key, reference_tensor in reference.items():
        if reference_tensor.is_floating_point() or reference_tensor.is_complex():
            accumulator = torch.zeros_like(reference_tensor)
            for state_dict, count in validated:
                accumulator += state_dict[key] * (count / total_samples)
            result[key] = accumulator
        else:
            # Integer buffers (for example, batch counters) cannot be averaged.
            result[key] = reference_tensor.clone()

    return result  