import torch
import pytest

from model.network import SmallSegmentationModel
from model.train import train_one_step


def test_training_step_updates_parameters():
    torch.manual_seed(42)

    model = SmallSegmentationModel()
    images = torch.randn(2, 1, 128, 128)
    masks = torch.randint(0, 2, (2, 128, 128))

    parameters_before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    loss = train_one_step(model, images, masks)

    parameters_after = list(model.parameters())

    assert torch.isfinite(torch.tensor(loss))
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(parameters_before, parameters_after)
    )


def test_training_rejects_invalid_image_shape():
    model = SmallSegmentationModel()
    images = torch.randn(2, 128, 128)
    masks = torch.randint(0, 2, (2, 128, 128))

    with pytest.raises(ValueError, match="images must have shape"):
        train_one_step(model, images, masks)