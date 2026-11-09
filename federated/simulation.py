import torch
import numpy as np

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


class HospitalClient:
    def __init__(self, hospital_id: int):
        self.hospital_id = hospital_id
        self.model = SmallSegmentationModel()
        self.images, self.masks = create_hospital_data(hospital_id)

    def fit(self, global_parameters):
        set_model_parameters(self.model, global_parameters)

        loss = train_one_step(
            self.model, self.images, self.masks
        )

        print(
            f"Hospital {self.hospital_id}: "
            f"training loss = {loss:.4f}"
        )

        return (
            get_model_parameters(self.model),
            len(self.images),
            loss
        )

    def evaluate(self, global_parameters):
        set_model_parameters(self.model, global_parameters)
        self.model.eval()

        with torch.no_grad():
            predictions = self.model(self.images)
            loss = torch.nn.functional.cross_entropy(
                predictions, self.masks.long()
            )

        return float(loss.item())


def fedavg(updates):
    """Aggregate hospital parameters using sample-weighted FedAvg."""
    total_samples = sum(num_samples for _, num_samples, _ in updates)

    aggregated = []

    for parameter_group in zip(*(params for params, _, _ in updates)):
        weighted_parameter = sum(
            parameter * num_samples
            for parameter, (_, num_samples, _) in zip(
                parameter_group, updates
            )
        ) / total_samples

        aggregated.append(weighted_parameter)

    return aggregated


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    clients = [
        HospitalClient(i) for i in range(NUM_HOSPITALS)
    ]

    global_model = SmallSegmentationModel()
    global_parameters = get_model_parameters(global_model)

    print("\nStarting FEDMED local federated-learning simulation...")
    print(f"Simulated hospitals: {NUM_HOSPITALS}")
    print(f"Federated rounds: {NUM_ROUNDS}")

    for round_number in range(1, NUM_ROUNDS + 1):
        print(f"\n========== ROUND {round_number} ==========")

        updates = []

        for client in clients:
            update = client.fit(global_parameters)
            updates.append(update)

        global_parameters = fedavg(updates)
        set_model_parameters(global_model, global_parameters)

        print("Global model aggregation completed.")

        evaluation_losses = [
            client.evaluate(global_parameters)
            for client in clients
        ]

        average_loss = sum(evaluation_losses) / len(evaluation_losses)

        print(
            f"Round {round_number} average evaluation loss: "
            f"{average_loss:.4f}"
        )

    print("\nFEDMED federated-learning simulation completed successfully.")


if __name__ == "__main__":
    main()