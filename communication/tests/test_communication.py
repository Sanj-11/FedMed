import grpc
import sys
from pathlib import Path
import pytest

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





