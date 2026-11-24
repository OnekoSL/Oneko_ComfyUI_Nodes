import hashlib
import math
import re
import struct
from contextlib import ExitStack
from functools import partial

import torch

from comfy.ldm.aura.mmdit import MMDiT as AuraFlowDiT
from comfy.patcher_extension import WrappersMP
from comfy_api.latest import io

from .block_noise_core import add_feature_noise, _sigma_is_active, _sigma_value
from .noise_sampler_core import NOISE_TYPES, generate_oneko_noise_for_tensor

try:
    from comfy.ldm.anima.model import Anima
except ImportError:
    Anima = None  # Older Core versions can still use the other supported models.


_MAX_SEED = 0xffffffffffffffff
_ANIMA_NOISE_CALLS = "oneko_anima_block_noise_calls"


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
            raise ValueError(f"Block range '{part.strip()}' is invalid for this model. Valid indices are 0-{block_count - 1}.")
        selected.update(range(first, last + 1))
    return tuple(sorted(selected))


def _noise_seed(seed, block_index, sigma):
    digest = hashlib.blake2b(digest_size=8, person=b"OnekoDiT")
    digest.update(struct.pack("<Q", int(seed) & _MAX_SEED))
    digest.update(struct.pack("<I", block_index))
    digest.update(struct.pack("<d", float(sigma)))
    return int.from_bytes(digest.digest(), "little", signed=False)


def _block_layout(diffusion):
    if isinstance(diffusion, AuraFlowDiT):
        double_blocks = [("double_block", index) for index in range(len(diffusion.double_layers))]
        single_blocks = [("single_block", index) for index in range(len(diffusion.single_layers))]
        return double_blocks + single_blocks, diffusion.patch_size, True

    # AuraFlow and the rest of Oneko can run on Cores predating MOSS support.
    try:
        from comfy.ldm.moss_image.model import MOSSImage
    except ImportError as exc:
        raise ValueError(
            "DiT Block Noise supports AuraFlow (Pony V7) and Anima on compatible Core versions. "
            "MOSS requires a ComfyUI version with native MOSS support."
        ) from exc
    if isinstance(diffusion, MOSSImage):
        return [("single_block", index) for index in range(len(diffusion.layers))], 1, False
    raise ValueError(
        f"DiT Block Noise supports MOSS, AuraFlow (Pony V7), and Anima. Received {type(diffusion).__name__}. "
        "For SD1/SD2/SDXL use UNet Block Noise."
    )


def _noisy_block(previous, block_index, height, width, sigma, seed, noise_type, strength, block_scale,
                 *, image_at_end=False):
    image_tokens = height * width
    image_slice = slice(-image_tokens, None) if image_at_end else slice(0, image_tokens)

    def replacement(args, extra_args):
        original_block = extra_args["original_block"]

        def original_with_noise(block_args):
            result = original_block(block_args)
            tokens = result["img"]
            # MOSS places the image first. AuraFlow's single blocks append it
            # after register and text tokens; its double-block img is all image.
            target = tokens[:, image_slice]
            image = target.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], height, width)
            noise = generate_oneko_noise_for_tensor(image, _noise_seed(seed, block_index, sigma), noise_type, 1.0)
            image = add_feature_noise(image, noise, strength, block_scale)
            # MOSS keeps this buffer outside its memory-compiler block scopes.
            # Preserve its storage: a new output tensor would escape the scope.
            target.copy_(image.flatten(2).transpose(1, 2))
            return result

        # SLG and other full replacements can skip the original block entirely.
        # In that case they also skip this block's noise.
        if previous is not None:
            return previous(args, {**extra_args, "original_block": original_with_noise})
        return original_with_noise(args)

    return replacement


def _anima_noise_wrapper(selected, seed, noise_type, strength, block_scale, sigma_start, sigma_end):
    sigma_low, sigma_high = sorted((sigma_start, sigma_end))

    def diffusion_wrapper(executor, *args, transformer_options=None, **kwargs):
        options = {} if transformer_options is None else transformer_options
        sigma = _sigma_value(options)
        if not sigma_low <= sigma <= sigma_high:
            return executor(*args, transformer_options=options, **kwargs)

        # The current Core calls Anima's blocks directly. Scope the hooks to
        # this forward and tag its options so other calls on the model skip them.
        call = object()
        options = options.copy()
        options[_ANIMA_NOISE_CALLS] = (*options.get(_ANIMA_NOISE_CALLS, ()), call)

        def add_noise(index, _module, _inputs, block_kwargs, output=None):
            # A concurrent forward can hold a snapshot of an already removed hook.
            if output is None or call not in block_kwargs.get("transformer_options", {}).get(_ANIMA_NOISE_CALLS, ()):
                return None
            # Anima's residual is B,T,H,W,C. Noise profiles use the final H,W axes.
            image = output.movedim(-1, 1)
            noise = generate_oneko_noise_for_tensor(image, _noise_seed(seed, index, sigma), noise_type, 1.0)
            return add_feature_noise(image, noise, strength, block_scale).movedim(1, -1)

        with ExitStack() as hooks:
            for index in selected:
                block = executor.class_obj.blocks[index]
                hooks.enter_context(block.register_forward_hook(partial(add_noise, index), with_kwargs=True))
            return executor(*args, transformer_options=options, **kwargs)

    return diffusion_wrapper


class OnekoDiTBlockNoisePatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OnekoDiTBlockNoisePatch",
            display_name="DiT Block Noise (Oneko)",
            category="Oneko/09 Model Patches",
            description=(
                "Add noise to image features after selected Anima, MOSS, or AuraFlow (Pony V7) transformer blocks. "
                "SLG skips also skip noise. Strength 0 is neutral."
            ),
            is_experimental=True,
            search_aliases=["Anima", "MOSS", "AuraFlow", "Pony V7", "DiT", "block noise"],
            inputs=[
                io.Model.Input("model"),
                io.String.Input(
                    "blocks", default="all",
                    tooltip="Zero-based indices, e.g. 0, 4-7, 22. Use all for every block or leave empty to disable. "
                            "AuraFlow counts double blocks first, followed by single blocks.",
                ),
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
        diffusion = model.get_model_object("diffusion_model")
        is_anima = Anima is not None and isinstance(diffusion, Anima)
        if is_anima:
            block_count = len(diffusion.blocks)
        else:
            block_keys, patch_size, image_at_end = _block_layout(diffusion)
            block_count = len(block_keys)
        selected = _parse_blocks(blocks, block_count)
        if noise_type not in NOISE_TYPES:
            raise ValueError(f"Unknown noise type: {noise_type}")
        patched = model.clone()
        strength = float(noise_strength)
        if not selected or strength == 0.0:
            return io.NodeOutput(patched)
        sampling = model.get_model_object("model_sampling")
        sigma_start = float(sampling.percent_to_sigma(start_percent))
        sigma_end = float(sampling.percent_to_sigma(end_percent))
        block_scale = 1.0 / math.sqrt(len(selected))

        if is_anima:
            wrapper = _anima_noise_wrapper(selected, seed, noise_type, strength, block_scale, sigma_start, sigma_end)
            patched.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "oneko_dit_block_noise", wrapper)
            return io.NodeOutput(patched)

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
            height, width = ((side + patch_size - 1) // patch_size for side in x.shape[-2:])
            for index in selected:
                key = block_keys[index]
                dit[key] = _noisy_block(dit.get(key), index, height, width, sigma, seed,
                                        noise_type, strength, block_scale, image_at_end=image_at_end)
            replacements["dit"] = dit
            options["patches_replace"] = replacements
            return executor(x, timestep, context, transformer_options=options, **kwargs)

        patched.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "oneko_dit_block_noise", diffusion_wrapper)
        return io.NodeOutput(patched)


NODE_CLASS_MAPPINGS = {"OnekoDiTBlockNoisePatch": OnekoDiTBlockNoisePatch}
NODE_DISPLAY_NAME_MAPPINGS = {"OnekoDiTBlockNoisePatch": "DiT Block Noise (Oneko)"}
