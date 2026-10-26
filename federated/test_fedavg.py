import numpy as np
import pytest

from federated.fedavg import aggregate_parameters


def test_fedavg_uses_example_weighting():
    hospital_a = [np.array([1.0, 3.0], dtype=np.float32)]
    hospital_b = [np.array([5.0, 7.0], dtype=np.float32)]

    result = aggregate_parameters(
        [hospital_a, hospital_b],
        [1, 3],
    )

    np.testing.assert_allclose(
        result[0],
        np.array([4.0, 6.0], dtype=np.float32),
    )


def test_fedavg_rejects_empty_clients():
    with pytest.raises(ValueError, match="must not be empty"):
        aggregate_parameters([], [])


def test_fedavg_rejects_mismatched_client_counts():
    with pytest.raises(ValueError, match="corresponding example count"):
        aggregate_parameters([[np.array([1.0])]], [])


def test_fedavg_rejects_nonpositive_example_counts():
    with pytest.raises(ValueError, match="positive integer"):
        aggregate_parameters([[np.array([1.0])]], [0])


def test_fedavg_rejects_different_parameter_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        aggregate_parameters(
            [[np.array([1.0, 2.0])], [np.array([3.0])]],
            [1, 1],
        )