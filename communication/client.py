import logging
import os

import grpc

import hospital_pb2
import hospital_pb2_grpc


SERVER_ADDRESS = "127.0.0.1:50051"
DEFAULT_REQUEST_TIMEOUT = 5.0


def get_request_timeout():
    """Return a valid positive RPC timeout from the environment."""
    raw_timeout = os.getenv(
        "FEDMED_REQUEST_TIMEOUT",
        str(DEFAULT_REQUEST_TIMEOUT),
    )

    try:
        timeout = float(raw_timeout)
    except (TypeError, ValueError):
        logging.warning(
            "Invalid FEDMED_REQUEST_TIMEOUT=%r; using %.1f seconds.",
            raw_timeout,
            DEFAULT_REQUEST_TIMEOUT,
        )
        return DEFAULT_REQUEST_TIMEOUT

    if timeout <= 0:
        logging.warning(
            "FEDMED_REQUEST_TIMEOUT must be positive; using %.1f seconds.",
            DEFAULT_REQUEST_TIMEOUT,
        )
        return DEFAULT_REQUEST_TIMEOUT

    return timeout


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    request_timeout = get_request_timeout()

    try:
        with grpc.insecure_channel(SERVER_ADDRESS) as channel:
            stub = hospital_pb2_grpc.FederatedCommunicationStub(channel)

            try:
                response = stub.RegisterHospital(
                    hospital_pb2.RegisterHospitalRequest(
                        hospital_id="hospital_test_01"
                    ),
                    timeout=request_timeout,
                )

                print("Registration response received!")
                print(f"Accepted: {response.accepted}")
                print(f"Message: {response.message}")

            except grpc.RpcError as error:
                logging.error(
                    "Hospital registration failed: %s - %s",
                    error.code().name,
                    error.details(),
                )
                return

            try:
                update_response = stub.SubmitModelUpdate(
                    hospital_pb2.ModelUpdateRequest(
                        hospital_id="hospital_test_01",
                        round_id="round_001",
                        model_update=b"test_model_update",
                        num_examples=10,
                    ),
                    timeout=request_timeout,
                )

                print("\nModel update response received!")
                print(f"Accepted: {update_response.accepted}")
                print(f"Message: {update_response.message}")

            except grpc.RpcError as error:
                logging.error(
                    "Model update submission failed: %s - %s",
                    error.code().name,
                    error.details(),
                )

    except (grpc.RpcError, OSError) as error:
        logging.error("Could not communicate with the server: %s", error)


if __name__ == "__main__":
    main()