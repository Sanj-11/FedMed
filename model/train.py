import torch
import torch.nn as nn
import torch.optim as optim

from model.network import SmallSegmentationModel


def train_one_step(
    model: nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    learning_rate: float = 0.001,
) -> float:
    """Train a segmentation model for one batch and return its loss."""

    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channels, height, width]")

    if masks.ndim != 3:
        raise ValueError("masks must have shape [batch, height, width]")

    if images.shape[0] != masks.shape[0] or images.shape[2:] != masks.shape[1:]:
        raise ValueError("images and masks must have matching batch and spatial dimensions")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()
    predictions = model(images)
    loss = criterion(predictions, masks.long())

    loss.backward()
    optimizer.step()

    return float(loss.detach().item())


def main() -> None:
    torch.manual_seed(42)

    model = SmallSegmentationModel()
    images = torch.randn(2, 1, 128, 128)
    masks = torch.randint(0, 2, (2, 128, 128))

    loss = train_one_step(model, images, masks)
    print(f"Training step completed. Loss: {loss:.4f}")


if __name__ == "__main__":
    main()