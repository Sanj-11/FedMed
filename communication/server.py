from concurrent import futures
import logging

import grpc

import hospital_pb2
import hospital_pb2_grpc


class FederatedCommunicationService(
    hospital_pb2_grpc.FederatedCommunicationServicer
):
    def RegisterHospital(self, request, context):
        hospital_id = request.hospital_id.strip()

        if not hospital_id:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "hospital_id must not be empty",
            )

        logging.info("Hospital registered: %s", hospital_id)

        return hospital_pb2.RegisterHospitalResponse(
            accepted=True,
            message=f"Hospital {hospital_id} registered",
        )

    def GetTrainingInstructions(self, request, context):
        if not request.hospital_id.strip():
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "hospital_id must not be empty",
            )

        # Placeholder until the team's training coordinator is integrated.
        return hospital_pb2.TrainingInstructionsResponse(
            training_available=False,
            round_id="",
            model_version="",
        )

    def SubmitModelUpdate(self, request, context):
        if not request.hospital_id.strip():
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "hospital_id must not be empty",
            )

        if not request.round_id.strip():
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "round_id must not be empty",
            )

        if not request.model_update:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "model_update must not be empty",
            )

        if request.num_examples == 0:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "num_examples must be greater than zero",
            )

        # Placeholder: validation and forwarding to the aggregation
        # component will be added during integration.
        return hospital_pb2.ModelUpdateResponse(
            accepted=False,
            message="Model update receiver is not integrated yet",
        )


def serve():
    logging.basicConfig(level=logging.INFO)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10)
    )

    hospital_pb2_grpc.add_FederatedCommunicationServicer_to_server(
        FederatedCommunicationService(),
        server,
    )

    port = server.add_insecure_port("127.0.0.1:50051")
    if port == 0:
        raise RuntimeError("Could not bind gRPC server to port 50051")

    server.start()
    logging.info("Development gRPC server listening on 127.0.0.1:%s", port)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Stopping gRPC server")
        server.stop(grace=3).wait()


if __name__ == "__main__":
    serve()