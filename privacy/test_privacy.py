import pytest
import torch
from privacy.differential_privacy import clip_and_add_gaussian_noise


def test_clipping_bounds_update_norm_without_noise():
    state={'w':torch.tensor([3.0,4.0])}
    result=clip_and_add_gaussian_noise(state,max_norm=1.0,noise_multiplier=0.0)
    assert torch.linalg.vector_norm(result['w']).item() == pytest.approx(1.0,abs=1e-5)


def test_rejects_nonfinite_values():
    with pytest.raises(ValueError,match='non-finite'):
        clip_and_add_gaussian_noise({'w':torch.tensor([float('nan')])})
