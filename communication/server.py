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
            logging.warning(
                "Rejected hospital registration: empty hospital_id"
            )
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
        hospital_id = request.hospital_id.strip()

        if not hospital_id:
            logging.warning(
                "Rejected training-instructions request: empty hospital_id"
            )
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
        hospital_id = request.hospital_id.strip()

        if not hospital_id:
            logging.warning(
                "Rejected model update request: empty hospital_id"
            )
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "hospital_id must not be empty",
            )

        if not request.round_id.strip():
            logging.warning(
                "Rejected model update request: empty round_id"
            )
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "round_id must not be empty",
            )

        if not request.model_update:
            logging.warning(
                "Rejected model update request: empty model_update"
            )
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "model_update must not be empty",
            )

        if request.num_examples == 0:
            logging.warning(
                "Rejected model update request: num_examples is zero"
            )
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "num_examples must be greater than zero",
            )

        # Acknowledge the update for the current development stage.
        # This does not store or aggregate the model update.
        logging.info(
            "Model update received from hospital %s for round %s "
            "with %s examples (%s bytes)",
            hospital_id,
            request.round_id,
            request.num_examples,
            len(request.model_update),
        )

        return hospital_pb2.ModelUpdateResponse(
            accepted=True,
            message="Model update received successfully",
        )


def serve():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10)
    )

    hospital_pb2_grpc.add_FederatedCommunicationServicer_to_server(
        FederatedCommunicationService(),
        server,
    )

    port = server.add_insecure_port("127.0.0.1:50051")

    if port == 0:
        raise RuntimeError(
            "Could not bind gRPC server to port 50051"
        )

    server.start()
    logging.info(
        "Development gRPC server listening on 127.0.0.1:%s",
        port,
    )

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Stopping gRPC server...")
        server.stop(grace=3).wait()
        logging.info("gRPC server stopped.")


if __name__ == "__main__":
    serve()