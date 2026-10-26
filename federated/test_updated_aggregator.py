import io

import pytest
import torch

from federated.update_aggregator import (
    aggregate_serialized_updates,
    decode_model_update,
)


def serialize_state(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()


def test_decode_model_update():
    payload = serialize_state({
        "weight": torch.tensor([1.0, 2.0]),
    })

    result = decode_model_update(payload)

    assert result[0].tolist() == [1.0, 2.0]


def test_aggregate_serialized_updates_weighted_fedavg():
    first = serialize_state({
        "weight": torch.tensor([1.0, 3.0]),
    })
    second = serialize_state({
        "weight": torch.tensor([5.0, 7.0]),
    })

    result = aggregate_serialized_updates([first, second], [1, 3])

    assert result[0].tolist() == [4.0, 6.0]


def test_decode_rejects_empty_payload():
    with pytest.raises(ValueError, match="non-empty bytes"):
        decode_model_update(b"")


def test_decode_rejects_invalid_payload():
    with pytest.raises(ValueError, match="Invalid serialized"):
        decode_model_update(b"not a valid model")


def test_aggregate_rejects_empty_updates():
    with pytest.raises(ValueError, match="must not be empty"):
        aggregate_serialized_updates([], [])


def test_aggregate_rejects_mismatched_counts():
    payload = serialize_state({
        "weight": torch.tensor([1.0]),
    })

    with pytest.raises(ValueError, match="corresponding example count"):
        aggregate_serialized_updates([payload, payload], [5])


def test_aggregate_rejects_mismatched_shapes():
    first = serialize_state({
        "weight": torch.tensor([1.0, 2.0]),
    })
    second = serialize_state({
        "weight": torch.tensor([3.0]),
    })

    with pytest.raises(ValueError, match="shape mismatch"):
        aggregate_serialized_updates([first, second], [2, 2])


def test_aggregate_rejects_zero_examples():
    payload = serialize_state({
        "weight": torch.tensor([1.0]),
    })

    with pytest.raises(ValueError, match="positive integer"):
        aggregate_serialized_updates([payload], [0])


def test_aggregate_rejects_mismatched_parameter_names():
    first = serialize_state({
        "weight": torch.tensor([1.0, 2.0]),
    })
    second = serialize_state({
        "bias": torch.tensor([3.0, 4.0]),
    })

    with pytest.raises(ValueError, match="parameter names"):
        aggregate_serialized_updates([first, second], [2, 2])