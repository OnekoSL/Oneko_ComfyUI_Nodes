"""Shared feature-noise scaling and denoising-window helpers."""

import torch


def _sigma_is_active(transformer_options, sigma_start, sigma_end):
    sigma = _sigma_value(transformer_options)
    low = min(float(sigma_start), float(sigma_end))
    high = max(float(sigma_start), float(sigma_end))
    return low <= sigma <= high


def _sigma_value(transformer_options):
    sigmas = transformer_options.get("sigmas")
    if sigmas is None:
        return 0.0
    if isinstance(sigmas, torch.Tensor):
        if sigmas.numel() == 0:
            return 0.0
        return float(sigmas.reshape(-1)[0].detach().cpu().item())
    if isinstance(sigmas, (list, tuple)):
        return float(sigmas[0]) if sigmas else 0.0
    return float(sigmas)


def _feature_magnitude(tensor):
    tensor_float = tensor.float()
    if tensor_float.ndim <= 1:
        magnitude = tensor_float.square().mean().sqrt()
        return magnitude.clamp_min(1e-6)

    dims = tuple(range(1, tensor_float.ndim))
    magnitude = tensor_float.square().mean(dim=dims, keepdim=True).sqrt()
    return magnitude.clamp_min(1e-6)


def add_feature_noise(tensor, noise, noise_strength, group_scale=1.0):
    feature_scale = _feature_magnitude(tensor)
    noise = noise.to(dtype=feature_scale.dtype, device=tensor.device)
    noise = noise * feature_scale * float(noise_strength) * float(group_scale)
    return tensor + noise.to(dtype=tensor.dtype)
