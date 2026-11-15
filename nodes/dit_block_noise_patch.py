import hashlib
import math
import re
import struct

import torch

from comfy.patcher_extension import WrappersMP
from comfy_api.latest import io

from .block_noise_core import add_feature_noise, _sigma_is_active, _sigma_value
from .noise_sampler_core import NOISE_TYPES, generate_oneko_noise_for_tensor


_MAX_SEED = 0xffffffffffffffff


def _parse_blocks(text, block_count):
    text = text.strip()
    if text.lower() == "all":
        return tuple(range(block_count))
    if not text:
        return ()
    selected = set()
    for part in text.split(","):
        match = re.fullmatch(r"\s*(\d+)(?:\s*-\s*(\d+))?\s*", part)
        if match is None:
            raise ValueError("Blocks must be 'all', empty, or comma-separated indices/ranges such as '0, 4-7, 22'.")
        first = int(match[1])
        last = int(match[2]) if match[2] is not None else first
        if first > last or last >= block_count:
            raise ValueError(f"Block range '{part.strip()}' is invalid for this MOSS model; valid indices are 0-{block_count - 1}.")
        selected.update(range(first, last + 1))
    return tuple(sorted(selected))


def _noise_seed(seed, block_index, sigma):
    digest = hashlib.blake2b(digest_size=8, person=b"OnekoDiT")
    digest.update(struct.pack("<Q", int(seed) & _MAX_SEED))
    digest.update(struct.pack("<I", block_index))
    digest.update(struct.pack("<d", float(sigma)))
    return int.from_bytes(digest.digest(), "little", signed=False)


def _noisy_block(previous, block_index, height, width, sigma, seed, noise_type, strength, block_scale):
    def replacement(args, extra_args):
        original_block = extra_args["original_block"]

        def original_with_noise(block_args):
            result = original_block(block_args)
            tokens = result["img"]
            image_tokens = height * width
            # MOSS concatenates [target image, text, references]. Noise profiles
            # require the target's spatial layout, not a generic B,T,C tensor.
            image = tokens[:, :image_tokens].transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], height, width)
            noise = generate_oneko_noise_for_tensor(image, _noise_seed(seed, block_index, sigma), noise_type, 1.0)
            image = add_feature_noise(image, noise, strength, block_scale)
            # MOSS keeps this buffer outside its memory-compiler block scopes.
            # Preserve its storage: a new output tensor would escape the scope.
            tokens[:, :image_tokens].copy_(image.flatten(2).transpose(1, 2))
            return result

        # SLG and other full replacements can skip the original block entirely.
        # In that case they also skip this block's noise.
        if previous is not None:
            return previous(args, {**extra_args, "original_block": original_with_noise})
        return original_with_noise(args)

    return replacement


class OnekoDiTBlockNoisePatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OnekoDiTBlockNoisePatch",
            display_name="DiT Block Noise (MOSS) (Oneko)",
            category="Oneko/09 Model Patches",
            description=(
                "Add noise to target-image tokens after selected MOSS transformer blocks. "
                "Text/reference tokens are not directly noised. SLG skips also skip noise. "
                "Currently supports MOSS only; strength 0 is neutral."
            ),
            is_experimental=True,
            search_aliases=["MOSS", "DiT", "block noise"],
            inputs=[
                io.Model.Input("model"),
                io.String.Input("blocks", default="all", tooltip="Zero-based block indices: all, 0, 4-7, 22. Empty disables the patch."),
                io.Combo.Input("noise_type", options=list(NOISE_TYPES), default="gaussian"),
                io.Float.Input("noise_strength", default=0.0, min=-5.0, max=5.0, step=0.001,
                               tooltip="Noise relative to image-feature magnitude, divided by sqrt(number of selected blocks). Negative values reverse the noise direction; 0 is neutral."),
                io.Int.Input("seed", default=0, min=0, max=_MAX_SEED, control_after_generate=True),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.001),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.001),
            ],
            outputs=[io.Model.Output("model")],
        )

    @classmethod
    def execute(cls, model, blocks="all", noise_type="gaussian", noise_strength=0.0,
                seed=0, start_percent=0.0, end_percent=1.0):
        # Keep the rest of Oneko importable on Cores predating MOSS support.
        try:
            from comfy.ldm.moss_image.model import MOSSImage
        except ImportError as exc:
            raise ValueError(
                "DiT Block Noise requires ComfyUI with native MOSS support. "
                "Update ComfyUI before using this node."
            ) from exc
        diffusion = model.get_model_object("diffusion_model")
        if not isinstance(diffusion, MOSSImage):
            raise ValueError(
                f"DiT Block Noise currently supports MOSS only, not {type(diffusion).__name__}. "
                "For SD1/SD2/SDXL use UNet Block Noise."
            )
        selected = _parse_blocks(blocks, len(diffusion.layers))
        if noise_type not in NOISE_TYPES:
            raise ValueError(f"Unknown noise type: {noise_type}")
        patched = model.clone()
        if not selected or float(noise_strength) == 0.0:
            return io.NodeOutput(patched)
        sampling = model.get_model_object("model_sampling")
        sigma_start = float(sampling.percent_to_sigma(start_percent))
        sigma_end = float(sampling.percent_to_sigma(end_percent))
        block_scale = 1.0 / math.sqrt(len(selected))

        def diffusion_wrapper(executor, x, timestep, context, transformer_options=None, **kwargs):
            options = {} if transformer_options is None else transformer_options
            if not _sigma_is_active(options, sigma_start, sigma_end):
                return executor(x, timestep, context, transformer_options=options, **kwargs)
            # Build replacements per forward, after SLG has chosen its runtime
            # skips. Copies keep shared model/sampler options unchanged.
            options = options.copy()
            replacements = options.get("patches_replace", {}).copy()
            dit = replacements.get("dit", {}).copy()
            sigma = _sigma_value(options)
            height, width = x.shape[-2:]
            for index in selected:
                key = ("single_block", index)
                dit[key] = _noisy_block(dit.get(key), index, height, width, sigma, seed,
                                        noise_type, float(noise_strength), block_scale)
            replacements["dit"] = dit
            options["patches_replace"] = replacements
            return executor(x, timestep, context, transformer_options=options, **kwargs)

        patched.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "oneko_dit_block_noise", diffusion_wrapper)
        return io.NodeOutput(patched)


NODE_CLASS_MAPPINGS = {"OnekoDiTBlockNoisePatch": OnekoDiTBlockNoisePatch}
NODE_DISPLAY_NAME_MAPPINGS = {"OnekoDiTBlockNoisePatch": "DiT Block Noise (MOSS) (Oneko)"}
