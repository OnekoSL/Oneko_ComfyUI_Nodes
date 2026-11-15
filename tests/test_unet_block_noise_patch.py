import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
from comfy.model_patcher import ModelPatcher
from Oneko_ComfyUI_Nodes.nodes import unet_block_noise_patch as block_noise


class MergeSkip(torch.nn.Module):
    def forward(self, x):
        hidden, skip = x.chunk(2, dim=1)
        return (hidden + skip) * 0.5


def native_unet(model_class=UNetModel):
    # Exercise Core's real forward and hooks with tiny, weight-free block stages.
    model = model_class.__new__(model_class)
    torch.nn.Module.__init__(model)
    model.model_channels = 4
    model.num_classes = None
    model.default_num_video_frames = None
    model.predict_codebook_ids = False
    model.time_embed = torch.nn.Identity()
    model.input_blocks = torch.nn.ModuleList([
        torch.nn.Sequential(torch.nn.Identity()),
        torch.nn.Sequential(torch.nn.Identity()),
    ])
    model.middle_block = torch.nn.Sequential(torch.nn.Identity())
    model.output_blocks = torch.nn.ModuleList([
        torch.nn.Sequential(MergeSkip()),
        torch.nn.Sequential(MergeSkip()),
    ])
    model.out = torch.nn.Identity()
    return model


def make_patcher(diffusion_model):
    host = torch.nn.Module()
    host.diffusion_model = diffusion_model
    host.model_sampling = SimpleNamespace(percent_to_sigma=lambda percent: 1.0 - percent)
    return ModelPatcher(host, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))


def apply_noise(model, strengths=(0.2, 0.3, 0.4), seeds=(11, 22, 33), start=0.0, end=1.0):
    return block_noise.OnekoUNetBlockNoisePatch().patch(
        model, start, end,
        "gaussian", strengths[0], seeds[0],
        "gaussian", strengths[1], seeds[1],
        "gaussian", strengths[2], seeds[2],
    )[0]


def run_unet(patcher, sigma=0.5):
    sample = torch.linspace(-1.0, 1.0, 2 * 4 * 8 * 8).reshape(2, 4, 8, 8)
    before = sample.clone()
    options = copy.deepcopy(patcher.model_options["transformer_options"])
    options["sigmas"] = torch.tensor([sigma])
    result = patcher.get_model_object("diffusion_model")(
        sample, timesteps=torch.ones(2), transformer_options=options,
    )
    torch.testing.assert_close(sample, before, rtol=0, atol=0)
    return result


@pytest.mark.parametrize("architecture", ["moss", "flux", "duck_unet"])
@pytest.mark.parametrize("strength", [0.0, 0.3], ids=["zero-strength", "active"])
def test_dit_or_lookalike_is_rejected_before_cloning(architecture, strength):
    if architecture == "moss":
        from comfy.ldm.moss_image.model import MOSSImage
        model_type = MOSSImage
    elif architecture == "flux":
        from comfy.ldm.flux.model import Flux
        model_type = Flux
    else:
        model_type = type("UNetLookalike", (torch.nn.Module,), {})
    diffusion = model_type.__new__(model_type)
    torch.nn.Module.__init__(diffusion)
    # Similar attributes and a full ModelPatcher API are not evidence of UNet hooks.
    diffusion.input_blocks = torch.nn.ModuleList([torch.nn.Identity()])
    diffusion.output_blocks = torch.nn.ModuleList([torch.nn.Identity()])
    model = make_patcher(diffusion)
    before = copy.deepcopy(model.model_options)
    with patch.object(model, "clone", wraps=model.clone) as clone:
        with pytest.raises(ValueError, match="MOSS and other DiT models") as error:
            apply_noise(model, strengths=(strength,) * 3)
    assert type(diffusion).__name__ in str(error.value)
    clone.assert_not_called()
    assert model.model_options == before


def test_missing_diffusion_model_has_actionable_error():
    model = make_patcher(torch.nn.Identity())
    del model.model.diffusion_model
    with pytest.raises(ValueError, match="Core UNetModel diffusion_model"):
        apply_noise(model)


def test_missing_patcher_methods_keep_the_existing_api_error():
    with pytest.raises(ValueError, match="Model does not support UNet block noise patches: clone"):
        apply_noise(object())


def test_object_patch_controls_architecture_detection():
    from comfy.ldm.moss_image.model import MOSSImage

    model = make_patcher(native_unet())
    replacement = MOSSImage.__new__(MOSSImage)
    torch.nn.Module.__init__(replacement)
    model.add_object_patch("diffusion_model", replacement)
    with pytest.raises(ValueError, match="does not support MOSSImage"):
        apply_noise(model)
    assert model.get_model_object("diffusion_model") is replacement
    assert isinstance(model.model.diffusion_model, UNetModel)


def test_unet_subclass_keeps_supported_native_hooks():
    class CustomUNet(UNetModel):
        pass

    original = make_patcher(native_unet(CustomUNet))
    result = apply_noise(original)
    assert not torch.equal(run_unet(result), run_unet(original))


def test_zero_strength_is_neutral_for_supported_unet():
    original = make_patcher(native_unet())
    rng_before = torch.random.get_rng_state()
    with patch.object(block_noise, "generate_oneko_noise_for_tensor", side_effect=AssertionError("no noise expected")):
        result = apply_noise(original, strengths=(0.0,) * 3)
        torch.testing.assert_close(run_unet(result), run_unet(original), rtol=0, atol=0)
    assert result is not original
    assert result.model is original.model
    assert original.model_options == {"transformer_options": {}}
    assert torch.equal(torch.random.get_rng_state(), rng_before)


@pytest.mark.parametrize("group,expected_calls", [(0, 2), (1, 1), (2, 2)], ids=["input", "middle", "output"])
def test_native_forward_calls_selected_noise_group_and_preserves_rng(group, expected_calls):
    model = make_patcher(native_unet())
    strengths = [0.0, 0.0, 0.0]
    strengths[group] = 0.35
    rng_before = torch.random.get_rng_state()
    with patch.object(block_noise, "generate_oneko_noise_for_tensor", wraps=block_noise.generate_oneko_noise_for_tensor) as generate:
        result = run_unet(apply_noise(model, strengths=strengths))
    assert generate.call_count == expected_calls
    assert torch.isfinite(result).all()
    assert not torch.equal(result, run_unet(model))
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert model.model_options == {"transformer_options": {}}


def test_seed_repeatability_changes_with_seed_and_sigma():
    original = make_patcher(native_unet())
    first = apply_noise(original)
    repeat = apply_noise(original)
    other_seed = apply_noise(original, seeds=(12, 23, 34))
    torch.testing.assert_close(run_unet(first), run_unet(repeat), rtol=0, atol=0)
    assert not torch.equal(run_unet(first), run_unet(other_seed))
    assert not torch.equal(run_unet(first, sigma=0.5), run_unet(first, sigma=0.6))


@pytest.mark.parametrize("start,end", [(0.2, 0.8), (0.8, 0.2)])
def test_native_forward_respects_denoising_window(start, end):
    original = make_patcher(native_unet())
    patched = apply_noise(original, start=start, end=end)
    torch.testing.assert_close(run_unet(patched, sigma=0.1), run_unet(original), rtol=0, atol=0)
    torch.testing.assert_close(run_unet(patched, sigma=0.9), run_unet(original), rtol=0, atol=0)
    assert not torch.equal(run_unet(patched, sigma=0.5), run_unet(original))


def test_existing_hooks_are_composed_without_changing_original_lists():
    original = make_patcher(native_unet())
    seen = []

    def existing_input(h, options):
        seen.append(options["block"])
        return h + 0.1

    original.set_model_input_block_patch(existing_input)
    old_options = copy.deepcopy(original.model_options)
    result = apply_noise(original, strengths=(0.35, 0.0, 0.0))
    assert original.model_options == old_options
    old_hooks = original.model_options["transformer_options"]["patches"]["input_block_patch"]
    new_hooks = result.model_options["transformer_options"]["patches"]["input_block_patch"]
    assert len(old_hooks) == 1 and len(new_hooks) == 2
    assert new_hooks[0] is old_hooks[0] is existing_input
    run_unet(result)
    assert seen == [("input", 0), ("input", 1)]


@pytest.mark.parametrize("group,expected", [
    ("input", 14363100055728174739),
    ("middle", 9027901746956280069),
    ("output", 2234392882727617524),
])
def test_existing_seed_personalization_is_stable(group, expected):
    assert block_noise._derive_noise_seed(123, group, 2, 0.5) == expected
