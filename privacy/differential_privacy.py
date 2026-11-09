"""Basic client-side Gaussian perturbation for model updates.

This utility is not, by itself, a formal DP guarantee: callers must clip the
whole update, account for sampling, and use a privacy accountant for epsilon.
"""
from collections.abc import Mapping
import torch


def clip_and_add_gaussian_noise(state: Mapping[str, torch.Tensor], *, max_norm: float = 1.0,
                                noise_multiplier: float = 0.1, generator=None):
    if max_norm <= 0 or noise_multiplier < 0:
        raise ValueError("max_norm must be positive and noise_multiplier non-negative")
    if not state:
        raise ValueError("state must not be empty")
    total_sq = torch.zeros((), dtype=torch.float64)
    for value in state.values():
        if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
            raise ValueError("all state values must be floating point tensors")
        if not torch.isfinite(value).all():
            raise ValueError("state contains non-finite values")
        total_sq += torch.sum(value.detach().double() ** 2)
    norm = torch.sqrt(total_sq)
    scale = min(1.0, max_norm / (float(norm) + 1e-12))
    return {k: (v.detach().clone() * scale + torch.randn(v.shape, dtype=v.dtype,
            device=v.device, generator=generator) * (noise_multiplier * max_norm))
            for k, v in state.items()}
