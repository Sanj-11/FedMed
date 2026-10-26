import torch
import flwr as fl

from model.network import SmallSegmentationModel
from model.train import train_one_step
from federated.parameters import get_model_parameters, set_model_parameters

NUM_HOSPITALS = 3
NUM_ROUNDS = 2


def create_hospital_data(seed: int):
    """Create synthetic data for one simulated hospital."""
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(2, 1, 128, 128, generator=generator)
    masks = torch.randint(0, 2, (2, 128, 128), generator=generator)
    return images, masks


class HospitalClient(fl.client.NumPyClient):
    def __init__(self, hospital_id: int):
        self.hospital_id = hospital_id
        self.model = SmallSegmentationModel()
        self.images, self.masks = create_hospital_data(hospital_id)

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        set_model_parameters(self.model, parameters)
        loss = train_one_step(self.model, self.images, self.masks)
        print(
            f"Hospital {self.hospital_id}: "
            f"training loss = {loss:.4f}"
        )
        return get_model_parameters(self.model), len(self.images), {"loss": loss}

    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self.images)
            loss = torch.nn.functional.cross_entropy(
                predictions, self.masks.long()
            )
        return float(loss.item()), len(self.images), {
            "hospital_id": self.hospital_id
        }


def client_fn(context: fl.common.Context):
    hospital_id = int(context.node_config["partition-id"])
    return HospitalClient(hospital_id).to_client()


def server_fn(context: fl.common.Context):
    initial_model = SmallSegmentationModel()
    initial_parameters = fl.common.ndarrays_to_parameters(
        get_model_parameters(initial_model)
    )

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_HOSPITALS,
        min_evaluate_clients=NUM_HOSPITALS,
        min_available_clients=NUM_HOSPITALS,
        initial_parameters=initial_parameters,
    )

    config = fl.server.ServerConfig(num_rounds=NUM_ROUNDS)
    return fl.server.ServerAppComponents(
        strategy=strategy,
        config=config,
    )


def main():
    torch.manual_seed(42)

    server_app = fl.server.ServerApp(server_fn=server_fn)
    client_app = fl.client.ClientApp(client_fn=client_fn)

    print("Starting FEDMED federated-learning simulation...")
    print(f"Simulated hospitals: {NUM_HOSPITALS}")
    print(f"Federated rounds: {NUM_ROUNDS}")

    fl.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=NUM_HOSPITALS,
        backend_config={"client_resources": {"num_cpus": 1}},
    )

    print("Federated-learning simulation completed.")


if __name__ == "__main__":
    main()