import grpc

import hospital_pb2
import hospital_pb2_grpc


def main():
    # Local development connection only; TLS will be added later.
    with grpc.insecure_channel("127.0.0.1:50051") as channel:
        stub = hospital_pb2_grpc.FederatedCommunicationStub(channel)

        try:
            response = stub.RegisterHospital(
                hospital_pb2.RegisterHospitalRequest(
                    hospital_id="hospital_test_01"
                ),
                timeout=5,
            )

            print("Registration response received!")
            print(f"Accepted: {response.accepted}")
            print(f"Message: {response.message}")

        except grpc.RpcError as error:
            print(f"gRPC error: {error.code().name}")
            print(f"Details: {error.details()}")


if __name__ == "__main__":
    main()