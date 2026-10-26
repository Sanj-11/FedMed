import grpc
import sys
from pathlib import Path
import pytest

import io
import torch



sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import client

import hospital_pb2
import hospital_pb2_grpc

def test_register_hospital():
    with grpc.insecure_channel("127.0.0.1:50051") as channel:
        stub = hospital_pb2_grpc.FederatedCommunicationStub(channel)

        response = stub.RegisterHospital(
            hospital_pb2.RegisterHospitalRequest(
                hospital_id="hospital_test_pytest"
            ),
            timeout=5,
        )

        assert response.accepted is True
        assert "registered" in response.message.lower()


def test_submit_model_update():
    with grpc.insecure_channel("127.0.0.1:50051") as channel:
        stub = hospital_pb2_grpc.FederatedCommunicationStub(channel)

        response = stub.SubmitModelUpdate(
            hospital_pb2.ModelUpdateRequest(
                hospital_id="hospital_test_pytest",
                round_id="round_pytest_001",
                model_update=b"test_model_update",
                num_examples=10,
            ),
            timeout=5,
        )

        assert response.accepted is True
        assert "successfully" in response.message.lower()

def test_register_hospital_empty_id():
    with grpc.insecure_channel("127.0.0.1:50051") as channel:
        stub = hospital_pb2_grpc.FederatedCommunicationStub(channel)

        with pytest.raises(grpc.RpcError) as exc:
            stub.RegisterHospital(
                hospital_pb2.RegisterHospitalRequest(
                    hospital_id=""
                ),
                timeout=5,
            )

        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_submit_model_update_zero_examples():
    with grpc.insecure_channel("127.0.0.1:50051") as channel:
        stub = hospital_pb2_grpc.FederatedCommunicationStub(channel)

        with pytest.raises(grpc.RpcError) as exc:
            stub.SubmitModelUpdate(
                hospital_pb2.ModelUpdateRequest(
                    hospital_id="hospital_test_pytest",
                    round_id="round_pytest_002",
                    model_update=b"test_model_update",
                    num_examples=0,
                ),
                timeout=5,
            )

        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_submit_model_update_negative_examples():
    with pytest.raises(ValueError, match="out of range"):
        hospital_pb2.ModelUpdateRequest(
            hospital_id="hospital_test_pytest",
            round_id="round_pytest_negative",
            model_update=b"test_model_update",
            num_examples=-1,
        )


def test_submit_model_update_empty_payload():
    with grpc.insecure_channel("127.0.0.1:50051") as channel:
        stub = hospital_pb2_grpc.FederatedCommunicationStub(channel)

        with pytest.raises(grpc.RpcError) as exc:
            stub.SubmitModelUpdate(
                hospital_pb2.ModelUpdateRequest(
                    hospital_id="hospital_test_pytest",
                    round_id="round_pytest_003",
                    model_update=b"",
                    num_examples=10,
                ),
                timeout=5,
            )

        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_submit_model_update_empty_hospital_id():
    with grpc.insecure_channel("127.0.0.1:50051") as channel:
        stub = hospital_pb2_grpc.FederatedCommunicationStub(channel)

        with pytest.raises(grpc.RpcError) as exc:
            stub.SubmitModelUpdate(
                hospital_pb2.ModelUpdateRequest(
                    hospital_id="",
                    round_id="round_pytest_004",
                    model_update=b"test_model_update",
                    num_examples=10,
                ),
                timeout=5,
            )

        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_submit_model_update_empty_round_id():
    with grpc.insecure_channel("127.0.0.1:50051") as channel:
        stub = hospital_pb2_grpc.FederatedCommunicationStub(channel)
        with pytest.raises(grpc.RpcError) as exc:
            stub.SubmitModelUpdate(
                hospital_pb2.ModelUpdateRequest(
                    hospital_id="hospital_test_pytest",
                    round_id="",
                    model_update=b"test_model_update",
                    num_examples=10,
                ),
                timeout=5,
            )
        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_get_training_instructions_empty_hospital_id():
    with grpc.insecure_channel("127.0.0.1:50051") as channel:
        stub = hospital_pb2_grpc.FederatedCommunicationStub(channel)
        with pytest.raises(grpc.RpcError) as exc:
            stub.GetTrainingInstructions(
                hospital_pb2.TrainingInstructionsRequest(
                    hospital_id=""
                ),
                timeout=5,
            )
        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_server_unavailable():
    with grpc.insecure_channel("127.0.0.1:50099") as channel:
        stub = hospital_pb2_grpc.FederatedCommunicationStub(channel)

        with pytest.raises(grpc.RpcError) as exc:
            stub.RegisterHospital(
                hospital_pb2.RegisterHospitalRequest(
                    hospital_id="hospital_unavailable_test"
                ),
                timeout=1,
            )

        assert exc.value.code() in (
            grpc.StatusCode.UNAVAILABLE,
            grpc.StatusCode.DEADLINE_EXCEEDED,
        )


def test_request_timeout_uses_default_when_unset(monkeypatch):
    monkeypatch.delenv("FEDMED_REQUEST_TIMEOUT", raising=False)

    assert client.get_request_timeout() == 5.0


def test_request_timeout_reads_valid_environment_value(monkeypatch):
    monkeypatch.setenv("FEDMED_REQUEST_TIMEOUT", "2")

    assert client.get_request_timeout() == 2.0


def test_request_timeout_falls_back_when_invalid(monkeypatch):
    monkeypatch.setenv("FEDMED_REQUEST_TIMEOUT", "abc")

    assert client.get_request_timeout() == 5.0


def test_request_timeout_falls_back_when_zero(monkeypatch):
    monkeypatch.setenv("FEDMED_REQUEST_TIMEOUT", "0")

    assert client.get_request_timeout() == 5.0


def test_request_timeout_falls_back_when_negative(monkeypatch):
    monkeypatch.setenv("FEDMED_REQUEST_TIMEOUT", "-3")

    assert client.get_request_timeout() == 5.0


def test_request_timeout_falls_back_when_infinite(monkeypatch):
    monkeypatch.setenv("FEDMED_REQUEST_TIMEOUT", "inf")

    assert client.get_request_timeout() == 5.0


def test_request_timeout_falls_back_when_nan(monkeypatch):
    monkeypatch.setenv("FEDMED_REQUEST_TIMEOUT", "nan")

    assert client.get_request_timeout() == 5.0



def test_client_handles_registration_rpc_failure(monkeypatch, caplog):
    class FakeRpcError(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.UNAVAILABLE

        def details(self):
            return "Server unavailable"

    class FakeStub:
        def RegisterHospital(self, request, timeout):
            raise FakeRpcError()

    class FakeChannel:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(
        client.grpc,
        "insecure_channel",
        lambda address: FakeChannel(),
    )
    monkeypatch.setattr(
        client.hospital_pb2_grpc,
        "FederatedCommunicationStub",
        lambda channel: FakeStub(),
    )

    client.main()

    assert "Hospital registration failed" in caplog.text
    assert "UNAVAILABLE" in caplog.text
    assert "Server unavailable" in caplog.text


def test_client_handles_model_update_rpc_failure(monkeypatch, caplog):
    class FakeRpcError(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.UNAVAILABLE

        def details(self):
            return "Server unavailable"

    class FakeStub:
        def RegisterHospital(self, request, timeout):
            return hospital_pb2.RegisterHospitalResponse(
                accepted=True,
                message="Hospital registered",
            )

        def SubmitModelUpdate(self, request, timeout):
            raise FakeRpcError()

    class FakeChannel:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(
        client.grpc,
        "insecure_channel",
        lambda address: FakeChannel(),
    )
    monkeypatch.setattr(
        client.hospital_pb2_grpc,
        "FederatedCommunicationStub",
        lambda channel: FakeStub(),
    )

    client.main()

    assert "Model update submission failed" in caplog.text
    assert "UNAVAILABLE" in caplog.text
    assert "Server unavailable" in caplog.text


def test_server_stores_first_update_without_aggregating():
    from communication.server import FederatedCommunicationService
    import io
    import torch
    from unittest.mock import Mock

    service = FederatedCommunicationService()
    context = Mock()

    buffer = io.BytesIO()
    torch.save(
        {"weight": torch.tensor([1.0, 3.0])},
        buffer,
    )

    request = hospital_pb2.ModelUpdateRequest(
        hospital_id="hospital_1",
        round_id="round_test",
        model_update=buffer.getvalue(),
        num_examples=1,
    )

    response = service.SubmitModelUpdate(request, context)

    assert response.accepted is True
    assert "hospital_1" in service.round_updates["round_test"]
    assert "round_test" not in service.aggregated_rounds  


def test_server_aggregates_after_second_hospital():
    import io
    import torch
    from unittest.mock import Mock
    from communication.server import FederatedCommunicationService

    service = FederatedCommunicationService()
    context = Mock()

    def make_request(hospital_id, weights, num_examples):
        buffer = io.BytesIO()
        torch.save(
            {"weight": torch.tensor(weights)},
            buffer,
        )
        return hospital_pb2.ModelUpdateRequest(
            hospital_id=hospital_id,
            round_id="round_aggregate_test",
            model_update=buffer.getvalue(),
            num_examples=num_examples,
        )

    first = make_request("hospital_1", [1.0, 3.0], 1)
    second = make_request("hospital_2", [5.0, 7.0], 3)

    response1 = service.SubmitModelUpdate(first, context)

    assert response1.accepted is True
    assert "round_aggregate_test" not in service.aggregated_rounds

    response2 = service.SubmitModelUpdate(second, context)

    assert response2.accepted is True
    assert (
        service.aggregated_rounds["round_aggregate_test"][0].tolist()
        == [4.0, 6.0]
    )      


def test_server_rejects_duplicate_hospital_update():
    import io
    import torch
    import grpc
    import pytest
    from unittest.mock import Mock
    from communication.server import FederatedCommunicationService

    service = FederatedCommunicationService()
    context = Mock()

    buffer = io.BytesIO()
    torch.save({"weight": torch.tensor([1.0, 3.0])}, buffer)

    request = hospital_pb2.ModelUpdateRequest(
        hospital_id="hospital_duplicate",
        round_id="round_duplicate_test",
        model_update=buffer.getvalue(),
        num_examples=2,
    )

    first_response = service.SubmitModelUpdate(request, context)
    assert first_response.accepted is True

    context.reset_mock()
    context.abort.side_effect = grpc.RpcError("Duplicate submission")

    with pytest.raises(grpc.RpcError, match="Duplicate submission"):
        service.SubmitModelUpdate(request, context)

    context.abort.assert_called_once_with(
        grpc.StatusCode.ALREADY_EXISTS,
        "This hospital has already submitted an update for this round",
    )    





