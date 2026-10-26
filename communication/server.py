from concurrent import futures
import logging
import threading

import grpc

from communication import hospital_pb2
from communication import hospital_pb2_grpc
from federated.update_aggregator import aggregate_serialized_updates


class FederatedCommunicationService(
    hospital_pb2_grpc.FederatedCommunicationServicer
):
    def __init__(self):
        # Updates are stored in memory, grouped by round and hospital.
        self.round_updates = {}
        self.aggregated_rounds = {}

        # Development setting: aggregate after two hospitals submit.
        self.min_hospitals_per_round = 2

        # Protect shared state from concurrent RPC requests.
        self.lock = threading.Lock()

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

        # Placeholder until the training coordinator is integrated.
        return hospital_pb2.TrainingInstructionsResponse(
            training_available=False,
            round_id="",
            model_version="",
        )

    def SubmitModelUpdate(self, request, context):
        hospital_id = request.hospital_id.strip()
        round_id = request.round_id.strip()

        if not hospital_id:
            logging.warning(
                "Rejected model update request: empty hospital_id"
            )
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "hospital_id must not be empty",
            )

        if not round_id:
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

        if request.num_examples <= 0:
            logging.warning(
                "Rejected model update request: num_examples is not positive"
            )
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "num_examples must be greater than zero",
            )

        payload = bytes(request.model_update)
        num_examples = request.num_examples

        with self.lock:
            # Do not accept further updates after a round is aggregated.
            if round_id in self.aggregated_rounds:
                logging.warning(
                    "Rejected update for already aggregated round %s",
                    round_id,
                )
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    "This round has already been aggregated",
                )

            round_updates = self.round_updates.setdefault(round_id, {})

            if hospital_id in round_updates:
                logging.warning(
                    "Rejected duplicate model update from hospital %s "
                    "for round %s",
                    hospital_id,
                    round_id,
                )
                context.abort(
                    grpc.StatusCode.ALREADY_EXISTS,
                    "This hospital has already submitted an update for this round",
                )

            # Store this hospital's serialized update and example count.
            round_updates[hospital_id] = (payload, num_examples)

            logging.info(
                "Stored model update from hospital %s for round %s "
                "(%s examples, %s bytes)",
                hospital_id,
                round_id,
                num_examples,
                len(payload),
            )

            # Wait until enough hospitals have submitted their updates.
            if len(round_updates) < self.min_hospitals_per_round:
                return hospital_pb2.ModelUpdateResponse(
                    accepted=True,
                    message="Model update received successfully",
                )

            updates = [
                item[0] for item in round_updates.values()
            ]
            example_counts = [
                item[1] for item in round_updates.values()
            ]

            try:
                aggregated_parameters = aggregate_serialized_updates(
                    updates,
                    example_counts,
                )
            except (ValueError, TypeError, RuntimeError) as exc:
                # Do not mark the round as aggregated if aggregation fails.
                round_updates.pop(hospital_id, None)

                logging.exception(
                    "Aggregation failed for round %s",
                    round_id,
                )
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"Could not aggregate model updates: {exc}",
                )

            # Retain the aggregate for subsequent integration work.
            self.aggregated_rounds[round_id] = aggregated_parameters

            logging.info(
                "Successfully aggregated round %s from %s hospitals",
                round_id,
                len(round_updates),
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