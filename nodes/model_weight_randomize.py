import torch

from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
from comfy.ops import CastWeightBiasOp
from comfy.quant_ops import QuantizedTensor
from comfy.utils import string_to_seed
from comfy.weight_adapter import WeightAdapterBase
from comfy_api.latest import io


try:
    from comfy.ldm.anima.model import Anima
except ImportError:
    Anima = None

try:
    from comfy.ldm.moss_image.model import MOSSImage
except ImportError:
    MOSSImage = None


UNET_BLOCK_GROUPS = ["input_blocks", "middle_block", "output_blocks"]
DIT_BLOCK_GROUPS = ["early_dit_blocks", "middle_dit_blocks", "late_dit_blocks"]
BLOCK_GROUPS = ["all", *UNET_BLOCK_GROUPS, *DIT_BLOCK_GROUPS]


def _weight_prefixes(diffusion, block_group):
    if block_group == "all":
        return ("",)
    if block_group in UNET_BLOCK_GROUPS:
        if not isinstance(diffusion, UNetModel):
            raise ValueError("Input, middle and output block groups require a Core UNetModel. For Anima or MOSS-Image, select a DiT block group.")
        return (f"{block_group}.",)
    if block_group not in DIT_BLOCK_GROUPS:
        raise ValueError(f"Unknown Model Weight Randomize block group: {block_group!r}.")

    if Anima is not None and isinstance(diffusion, Anima):
        path, block_count = "blocks", len(diffusion.blocks)
    elif MOSSImage is not None and isinstance(diffusion, MOSSImage):
        path, block_count = "layers", len(diffusion.layers)
    else:
        raise ValueError("DiT block groups require Anima or MOSS-Image. Use a UNet block group for SDXL or Pony XL, or select 'all'.")

    third = DIT_BLOCK_GROUPS.index(block_group)
    first = third * block_count // len(DIT_BLOCK_GROUPS)
    stop = (third + 1) * block_count // len(DIT_BLOCK_GROUPS)
    return tuple(f"{path}.{index}." for index in range(first, stop))


class _RandomWeightPatch(WeightAdapterBase):
    name = "oneko_random_weight"

    def __init__(self, loaded_keys, weights):
        # Core reconstructs adapters from these fields when prefetching patches.
        self.loaded_keys = loaded_keys
        self.weights = weights

    def passive_memory_usage(self):
        return 0

    def calculate_weight(self, weight, key, strength, strength_model, offset, function,
                         intermediate_dtype=torch.float32, original_weight=None):
        dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
        rng_device = weight.device if weight.device.type in ("cpu", "cuda", "xpu") else "cpu"
        seed = self.weights[0] ^ string_to_seed(key)
        generator = torch.Generator(device=rng_device).manual_seed(seed)
        random_weight = torch.randn(weight.shape, generator=generator, device=rng_device, dtype=dtype)
        random_weight = random_weight.to(weight.device)
        weight_norm = torch.linalg.vector_norm(weight, dtype=dtype)
        random_norm = torch.linalg.vector_norm(random_weight)
        random_weight.mul_(weight_norm / random_norm)
        if strength != 1.0:
            random_weight.mul_(strength).add_(weight, alpha=1.0 - strength)
        return random_weight.to(dtype=weight.dtype)


class OnekoModelWeightRandomize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OnekoModelWeightRandomize",
            display_name="Model Weight Randomize (Oneko)",
            category="Oneko/09 Model Patches",
            description="Blends diffusion model weight matrices and kernels with seeded random weights of the same norm.",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("strength", default=0.0, min=0.0, max=1.0, step=0.001,
                               tooltip="0 uses the input model. 1 replaces the selected weight tensors."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffff, control_after_generate=True),
                io.Combo.Input("block_group", options=BLOCK_GROUPS, default="all", optional=True,
                               socketless=True, tooltip="Select all weights, a UNet block group, or a third of the main Anima/MOSS-Image transformer stack."),
            ],
            outputs=[io.Model.Output("model")],
        )

    @classmethod
    def execute(cls, model, strength, seed, block_group="all"):
        if strength == 0.0:
            return io.NodeOutput(model)

        diffusion = model.get_model_object("diffusion_model")
        prefixes = _weight_prefixes(diffusion, block_group)
        dynamic = model.is_dynamic()
        adapter = _RandomWeightPatch((), (seed,))
        patches = {}
        for name, weight in diffusion.named_parameters():
            if not name.startswith(prefixes):
                continue
            module_name, _, param_name = name.rpartition(".")
            if param_name != "weight" or weight.ndim < 2 or weight.numel() == 0:
                continue
            if not (weight.is_floating_point() or isinstance(weight, QuantizedTensor)):
                continue
            if dynamic and not isinstance(diffusion.get_submodule(module_name), CastWeightBiasOp):
                continue
            patches[f"diffusion_model.{name}"] = adapter

        patched = model.clone()
        if not patched.add_patches(patches, strength_patch=strength):
            if block_group != "all":
                raise ValueError(f"Model Weight Randomize found no supported weights in block group {block_group!r}.")
            raise ValueError("Model Weight Randomize found no supported diffusion weight matrices or kernels.")
        return io.NodeOutput(patched)


NODE_CLASS_MAPPINGS = {"OnekoModelWeightRandomize": OnekoModelWeightRandomize}
NODE_DISPLAY_NAME_MAPPINGS = {"OnekoModelWeightRandomize": "Model Weight Randomize (Oneko)"}
