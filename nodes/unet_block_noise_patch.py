import hashlib
import math
import struct

import torch

from .noise_sampler_core import NOISE_TYPES, generate_oneko_noise_for_tensor


_MAX_SEED = 0xffffffffffffffff
_GROUP_CODES = {
    "input": 1,
    "middle": 2,
    "output": 3,
}


class OnekoUNetBlockNoisePatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "start_percent": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Denoising start for the block noise patch.",
                    },
                ),
                "end_percent": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Denoising end for the block noise patch.",
                    },
                ),
                "input_noise_type": (
                    NOISE_TYPES,
                    {
                        "default": "gaussian",
                        "tooltip": "Noise injected after each UNet input block.",
                    },
                ),
                "input_noise_strength": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": "Noise amount relative to the current input block feature magnitude.",
                    },
                ),
                "input_noise_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": _MAX_SEED,
                        "control_after_generate": True,
                    },
                ),
                "middle_noise_type": (
                    NOISE_TYPES,
                    {
                        "default": "gaussian",
                        "tooltip": "Noise injected after the UNet middle block.",
                    },
                ),
                "middle_noise_strength": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": "Noise amount relative to the current middle block feature magnitude.",
                    },
                ),
                "middle_noise_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": _MAX_SEED,
                        "control_after_generate": True,
                    },
                ),
                "output_noise_type": (
                    NOISE_TYPES,
                    {
                        "default": "gaussian",
                        "tooltip": "Noise injected into each UNet output block input.",
                    },
                ),
                "output_noise_strength": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": "Noise amount relative to the current output block feature magnitude.",
                    },
                ),
                "output_noise_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": _MAX_SEED,
                        "control_after_generate": True,
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Oneko/09 Model Patches"
    DESCRIPTION = "Experimental UNet patch that injects independent noise into input, middle, and output block groups."

    def patch(
        self,
        model,
        start_percent,
        end_percent,
        input_noise_type,
        input_noise_strength,
        input_noise_seed,
        middle_noise_type,
        middle_noise_strength,
        middle_noise_seed,
        output_noise_type,
        output_noise_strength,
        output_noise_seed,
    ):
        _require_block_patch_api(model)
        model_sampling = model.get_model_object("model_sampling")
        sigma_start = model_sampling.percent_to_sigma(start_percent)
        sigma_end = model_sampling.percent_to_sigma(end_percent)
        group_scales = _group_noise_scales(model)

        patched_model = model.clone()

        def input_block_patch(h, transformer_options):
            return _inject_block_noise(
                h,
                transformer_options,
                "input",
                input_noise_seed,
                input_noise_type,
                input_noise_strength,
                group_scales["input"],
                sigma_start,
                sigma_end,
            )

        def middle_block_patch(args):
            transformer_options = args["transformer_options"]
            out = args.copy()
            out["h"] = _inject_block_noise(
                args["h"],
                transformer_options,
                "middle",
                middle_noise_seed,
                middle_noise_type,
                middle_noise_strength,
                group_scales["middle"],
                sigma_start,
                sigma_end,
            )
            return out

        def output_block_patch(h, hsp, transformer_options):
            h = _inject_block_noise(
                h,
                transformer_options,
                "output",
                output_noise_seed,
                output_noise_type,
                output_noise_strength,
                group_scales["output"],
                sigma_start,
                sigma_end,
            )
            return h, hsp

        patched_model.set_model_input_block_patch(input_block_patch)
        patched_model.set_model_middle_block_after_patch(middle_block_patch)
        patched_model.set_model_output_block_patch(output_block_patch)
        return (patched_model,)


def _require_block_patch_api(model):
    missing = [
        name
        for name in (
            "clone",
            "get_model_object",
            "set_model_input_block_patch",
            "set_model_middle_block_after_patch",
            "set_model_output_block_patch",
        )
        if not callable(getattr(model, name, None))
    ]
    if missing:
        raise ValueError("Model does not support UNet block noise patches: {}".format(", ".join(missing)))


def _inject_block_noise(
    tensor,
    transformer_options,
    group,
    base_seed,
    noise_type,
    noise_strength,
    group_scale,
    sigma_start,
    sigma_end,
):
    if float(noise_strength) == 0.0 or not _sigma_is_active(transformer_options, sigma_start, sigma_end):
        return tensor

    seed = _derive_noise_seed(base_seed, group, _block_index(transformer_options), _sigma_value(transformer_options))
    noise = generate_oneko_noise_for_tensor(tensor, seed, noise_type, 1.0)
    feature_scale = _feature_magnitude(tensor)
    noise = noise.to(dtype=feature_scale.dtype, device=tensor.device)
    noise = noise * feature_scale * float(noise_strength) * float(group_scale)
    return tensor + noise.to(dtype=tensor.dtype)


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


def _block_index(transformer_options):
    block = transformer_options.get("block")
    if isinstance(block, (list, tuple)) and len(block) > 1:
        return int(block[1])
    return 0


def _feature_magnitude(tensor):
    tensor_float = tensor.float()
    if tensor_float.ndim <= 1:
        magnitude = tensor_float.square().mean().sqrt()
        return magnitude.clamp_min(1e-6)

    dims = tuple(range(1, tensor_float.ndim))
    magnitude = tensor_float.square().mean(dim=dims, keepdim=True).sqrt()
    return magnitude.clamp_min(1e-6)


def _group_noise_scales(model):
    return {
        "input": _repeated_group_scale(model, "diffusion_model.input_blocks"),
        "middle": 1.0,
        "output": _repeated_group_scale(model, "diffusion_model.output_blocks"),
    }


def _repeated_group_scale(model, block_path):
    try:
        block_count = len(model.get_model_object(block_path))
    except (AttributeError, KeyError, TypeError):
        block_count = 1
    return 1.0 / math.sqrt(max(1, block_count))


def _derive_noise_seed(base_seed, group, block_index, sigma):
    digest = hashlib.blake2b(digest_size=8, person=b"NukunUN")
    digest.update(struct.pack("<Q", int(base_seed) & _MAX_SEED))
    digest.update(struct.pack("<I", _GROUP_CODES[group]))
    digest.update(struct.pack("<q", int(block_index)))
    digest.update(struct.pack("<d", float(sigma)))
    return int.from_bytes(digest.digest(), "little", signed=False)


NODE_CLASS_MAPPINGS = {
    "OnekoUNetBlockNoisePatch": OnekoUNetBlockNoisePatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnekoUNetBlockNoisePatch": "UNet Block Noise Patch (Oneko)"
}
