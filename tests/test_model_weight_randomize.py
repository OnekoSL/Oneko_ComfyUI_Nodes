import json

import pytest
import torch

from comfy.ldm.anima.model import Anima
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
from comfy.ldm.moss_image.model import MOSSImage
from comfy.model_patcher import LowVramPatch, ModelPatcher
from comfy.ops import mixed_precision_ops
from comfy.quant_ops import QuantizedTensor
from Oneko_ComfyUI_Nodes.nodes.model_weight_randomize import OnekoModelWeightRandomize


CPU = torch.device("cpu")
ROOT_WEIGHT = "diffusion_model.weight"


def make_patcher(diffusion):
    host = torch.nn.Module()
    host.diffusion_model = diffusion
    host.other = torch.nn.Linear(6, 4)
    return ModelPatcher(host, load_device=CPU, offload_device=CPU)


def make_linear():
    layer = torch.nn.Linear(6, 4)
    with torch.no_grad():
        layer.weight.copy_(torch.linspace(-1.0, 1.0, 24).reshape(4, 6))
    return layer


def patched_weight(patcher, key=ROOT_WEIGHT):
    return patcher.patch_weight_to_device(key, device_to=CPU, return_weight=True)


def test_zero_strength_returns_original_patcher_without_changing_patches():
    original = make_patcher(make_linear())
    original.add_patches({ROOT_WEIGHT: (torch.ones(4, 6),)}, strength_patch=0.1)
    patch_uuid = original.patches_uuid

    result = OnekoModelWeightRandomize.execute(original, 0.0, 123)[0]

    assert result is original
    assert result.patches_uuid == patch_uuid
    assert len(result.patches[ROOT_WEIGHT]) == 1


def test_real_load_and_unpatch_only_touch_diffusion_matrix_weights():
    diffusion = torch.nn.Module()
    diffusion.proj = make_linear()
    diffusion.peer = make_linear()
    diffusion.conv = torch.nn.Conv2d(2, 3, 3)
    diffusion.norm = torch.nn.LayerNorm(6)
    diffusion.decoy = torch.nn.Module()
    diffusion.decoy.register_buffer("weight", torch.ones(4, 6))
    diffusion.decoy.matrix = torch.nn.Parameter(torch.ones(4, 6))
    original = make_patcher(diffusion)
    previous = {name: tensor.clone() for name, tensor in original.model.state_dict().items()}
    expected = {"diffusion_model.proj.weight", "diffusion_model.peer.weight", "diffusion_model.conv.weight"}
    probe = torch.ones(1, 6)
    before = diffusion.proj(probe).detach()

    randomized = OnekoModelWeightRandomize.execute(original, 1.0, 123)[0]

    assert randomized is not original
    assert not original.patches
    assert set(randomized.patches) == expected
    for name, tensor in original.model.state_dict().items():
        torch.testing.assert_close(tensor, previous[name], rtol=0, atol=0)

    try:
        randomized.patch_model(device_to=CPU)
        for name, tensor in randomized.model.state_dict().items():
            if name in expected:
                assert tensor.dtype == previous[name].dtype
                assert not torch.equal(tensor, previous[name])
                torch.testing.assert_close(tensor.norm(), previous[name].norm())
            else:
                torch.testing.assert_close(tensor, previous[name], rtol=0, atol=0)
        assert not torch.equal(diffusion.proj.weight, diffusion.peer.weight)
        assert not torch.equal(diffusion.proj(probe), before)
    finally:
        randomized.unpatch_model(device_to=CPU)

    for name, tensor in original.model.state_dict().items():
        torch.testing.assert_close(tensor, previous[name], rtol=0, atol=0)


def test_root_weight_seed_and_interpolation_include_prior_additive_patch():
    original = make_patcher(make_linear())
    delta = torch.full_like(original.model.diffusion_model.weight, 0.25)
    original.add_patches({ROOT_WEIGHT: (delta,)}, strength_patch=0.5)
    active_weight = patched_weight(original)
    rng_state = torch.random.get_rng_state()

    full = patched_weight(OnekoModelWeightRandomize.execute(original, 1.0, 41)[0])
    repeated = patched_weight(OnekoModelWeightRandomize.execute(original, 1.0, 41)[0])
    different = patched_weight(OnekoModelWeightRandomize.execute(original, 1.0, 42)[0])
    partial = patched_weight(OnekoModelWeightRandomize.execute(original, 0.25, 41)[0])

    torch.testing.assert_close(full.norm(), active_weight.norm())
    torch.testing.assert_close(full, repeated, rtol=0, atol=0)
    assert not torch.equal(full, different)
    torch.testing.assert_close(partial, active_weight.lerp(full, 0.25))
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert len(original.patches[ROOT_WEIGHT]) == 1


def test_low_vram_prefetch_preserves_randomization():
    original = make_patcher(make_linear())
    original.add_patches({ROOT_WEIGHT: (torch.ones(4, 6),)}, strength_patch=0.1)
    randomized = OnekoModelWeightRandomize.execute(original, 0.7, 42)[0]
    base_weight = original.model.diffusion_model.weight.detach().clone()
    reference = patched_weight(randomized)
    patch = LowVramPatch(ROOT_WEIGHT, randomized.patches)
    patch.prepare(None, None)

    result = patch(base_weight)

    torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_native_fp8_model_load_requantizes_and_unpatch_restores_original():
    ops = mixed_precision_ops(compute_dtype=torch.float32, full_precision_mm=True)
    diffusion = ops.Linear(16, 16, bias=False, device=CPU)
    config = json.dumps({"format": "float8_e4m3fn"}).encode("utf-8")
    diffusion.load_state_dict({
        "weight": torch.linspace(-2.0, 2.0, 256).reshape(16, 16).to(torch.float8_e4m3fn),
        "weight_scale": torch.tensor(0.5),
        "comfy_quant": torch.tensor(list(config), dtype=torch.uint8),
    })
    original = make_patcher(diffusion)
    before = diffusion.weight.dequantize().clone()
    randomized = OnekoModelWeightRandomize.execute(original, 1.0, 333)[0]

    try:
        randomized.patch_model(device_to=CPU)
        assert isinstance(diffusion.weight, QuantizedTensor)
        after = diffusion.weight.dequantize()
        assert not torch.equal(after, before)
        torch.testing.assert_close(after.norm(), before.norm(), rtol=0.03, atol=0)
    finally:
        randomized.unpatch_model(device_to=CPU)

    assert isinstance(diffusion.weight, QuantizedTensor)
    torch.testing.assert_close(diffusion.weight.dequantize(), before, rtol=0, atol=0)


@pytest.fixture
def unet_patcher():
    unet = UNetModel.__new__(UNetModel)
    torch.nn.Module.__init__(unet)
    unet.input_blocks = torch.nn.ModuleList([torch.nn.Sequential(make_linear(), torch.nn.LayerNorm(4))])
    unet.middle_block = torch.nn.Sequential(make_linear())
    unet.output_blocks = torch.nn.ModuleList([torch.nn.Sequential(make_linear())])
    unet.input_blocks_extra = make_linear()
    unet.middle_block_extra = make_linear()
    unet.output_blocks_extra = make_linear()
    unet.time_embed = make_linear()
    unet.out = make_linear()
    unet.other = torch.nn.Module()
    unet.other.input_blocks = make_linear()
    return make_patcher(unet)


def test_legacy_prompt_keeps_all_weights_as_the_default():
    inputs = OnekoModelWeightRandomize.INPUT_TYPES()
    assert set(inputs["required"]) == {"model", "strength", "seed"}
    _, group = inputs["optional"]["block_group"]
    assert group["options"] == [
        "all", "input_blocks", "middle_block", "output_blocks",
        "early_dit_blocks", "middle_dit_blocks", "late_dit_blocks",
    ]
    assert group["default"] == "all"
    original = make_patcher(make_linear())

    legacy = OnekoModelWeightRandomize.EXECUTE_NORMALIZED(model=original, strength=0.5, seed=42)[0]
    explicit = OnekoModelWeightRandomize.EXECUTE_NORMALIZED(
        model=original, strength=0.5, seed=42, block_group="all",
    )[0]

    assert set(legacy.patches) == set(explicit.patches) == {ROOT_WEIGHT}
    torch.testing.assert_close(patched_weight(legacy), patched_weight(explicit), rtol=0, atol=0)


@pytest.mark.parametrize(("group", "parameter"), [
    ("input_blocks", "0.0.weight"),
    ("middle_block", "0.weight"),
    ("output_blocks", "0.0.weight"),
])
def test_only_selected_unet_block_weights_receive_the_same_patch_as_all(unet_patcher, group, parameter):
    key = f"diffusion_model.{group}.{parameter}"
    selected = OnekoModelWeightRandomize.execute(unet_patcher, 0.5, 42, group)[0]
    all_weights = OnekoModelWeightRandomize.execute(unet_patcher, 0.5, 42, "all")[0]

    assert set(selected.patches) == {key}
    assert selected.patches.keys() < all_weights.patches.keys()
    torch.testing.assert_close(patched_weight(selected, key), patched_weight(all_weights, key), rtol=0, atol=0)


def test_blocks_reject_non_unet_but_zero_strength_still_passes_through():
    diffusion = torch.nn.Module()
    diffusion.input_blocks = torch.nn.ModuleList([torch.nn.Sequential(make_linear())])
    original = make_patcher(diffusion)

    assert OnekoModelWeightRandomize.execute(original, 0.0, 42, "input_blocks")[0] is original
    with pytest.raises(ValueError, match="UNet"):
        OnekoModelWeightRandomize.execute(original, 0.5, 42, "input_blocks")


def test_block_without_eligible_weights_is_reported(unet_patcher):
    unet_patcher.model.diffusion_model.middle_block = torch.nn.Sequential(torch.nn.LayerNorm(4))

    with pytest.raises(ValueError, match="middle_block"):
        OnekoModelWeightRandomize.execute(unet_patcher, 0.5, 42, "middle_block")


@pytest.mark.parametrize(("model_type", "block_path", "aux_path"), [
    (Anima, "blocks", "llm_adapter"),
    (MOSSImage, "layers", "text_embedder"),
])
def test_dit_groups_partition_only_main_blocks_and_match_all(model_type, block_path, aux_path):
    diffusion = model_type.__new__(model_type)
    torch.nn.Module.__init__(diffusion)
    setattr(diffusion, block_path, torch.nn.ModuleList([
        torch.nn.Sequential(make_linear(), torch.nn.LayerNorm(4)) for _ in range(11)
    ]))
    auxiliary = torch.nn.Module()
    auxiliary.blocks = torch.nn.ModuleList([torch.nn.Sequential(make_linear())])
    setattr(diffusion, aux_path, auxiliary)
    diffusion.head = make_linear()
    original = make_patcher(diffusion)
    all_weights = OnekoModelWeightRandomize.execute(original, 0.04, 42, "all")[0]
    changed = set()

    for group, indices in (
        ("early_dit_blocks", range(3)),
        ("middle_dit_blocks", range(3, 7)),
        ("late_dit_blocks", range(7, 11)),
    ):
        selected = OnekoModelWeightRandomize.execute(original, 0.04, 42, group)[0]
        keys = {f"diffusion_model.{block_path}.{i}.0.weight" for i in indices}
        assert set(selected.patches) == keys
        assert not changed.intersection(keys)
        changed.update(keys)
        for key in keys:
            torch.testing.assert_close(patched_weight(selected, key), patched_weight(all_weights, key), rtol=0, atol=0)

    assert set(all_weights.patches) - changed == {
        f"diffusion_model.{aux_path}.blocks.0.0.weight", "diffusion_model.head.weight",
    }
    assert not original.patches
    with pytest.raises(ValueError, match="UNet"):
        OnekoModelWeightRandomize.execute(original, 0.04, 42, "input_blocks")


def test_dit_group_rejects_unet_but_strength_zero_bypasses(unet_patcher):
    assert OnekoModelWeightRandomize.execute(unet_patcher, 0.0, 42, "early_dit_blocks")[0] is unet_patcher
    with pytest.raises(ValueError, match="Anima|MOSS|DiT"):
        OnekoModelWeightRandomize.execute(unet_patcher, 0.04, 42, "early_dit_blocks")
