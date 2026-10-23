import grpc
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

                  



       