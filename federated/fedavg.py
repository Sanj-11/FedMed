from collections.abc import Sequence

import numpy as np


def aggregate_parameters(
    client_parameters: Sequence[Sequence[np.ndarray]],
    num_examples: Sequence[int],
) -> list[np.ndarray]:
    """Aggregate client model parameters using weighted FedAvg."""

    if not client_parameters:
        raise ValueError("client_parameters must not be empty")

    if len(client_parameters) != len(num_examples):
        raise ValueError(
            "Each client's parameters must have a corresponding example count"
        )

    if any(not isinstance(count, (int, np.integer)) or isinstance(count, bool)
           or count <= 0 for count in num_examples):
        raise ValueError("Each example count must be a positive integer")

    parameter_count = len(client_parameters[0])
    if parameter_count == 0:
        raise ValueError("Client parameter lists must not be empty")

    for parameters in client_parameters:
        if len(parameters) != parameter_count:
            raise ValueError("Clients must provide the same number of arrays")

    total_examples = sum(num_examples)
    aggregated = []

    for layer_index in range(parameter_count):
        arrays = [
            np.asarray(parameters[layer_index])
            for parameters in client_parameters
        ]
        reference_shape = arrays[0].shape

        if any(array.shape != reference_shape for array in arrays):
            raise ValueError(
                f"Parameter shape mismatch at index {layer_index}"
            )

        if any(not np.issubdtype(array.dtype, np.number) for array in arrays):
            raise ValueError(
                f"Parameter at index {layer_index} must be numeric"
            )

        # Accumulate in float64 to reduce rounding error, then restore
        # the first client's parameter dtype.
        result = np.zeros(reference_shape, dtype=np.float64)

        for array, count in zip(arrays, num_examples, strict=True):
            result += array.astype(np.float64) * (count / total_examples)

        aggregated.append(result.astype(arrays[0].dtype))

    return aggregated