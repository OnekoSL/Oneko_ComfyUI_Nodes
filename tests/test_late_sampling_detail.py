"""Late detail must protect coarse frequencies and leave shared sampling state alone."""
import copy
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import comfy.samplers
from Oneko_ComfyUI_Nodes.nodes import noise_sampler_core as core
from Oneko_ComfyUI_Nodes.nodes.universal_noise_sampler import OnekoUniversalNoiseSamplerAdvanced
from .test_moss_sampling_integration import guider, inputs, moss_model


def test_detail_boost_preserves_coarse_spectrum_and_channel_means():
    source = torch.randn(2, 16, 32, 48, generator=torch.Generator().manual_seed(123))
    original = source.clone()
    actual = core._boost_spatial_detail(source, 0.12)
    before, after = torch.fft.rfft2(source), torch.fft.rfft2(actual)
    fy, fx = torch.fft.fftfreq(32), torch.fft.rfftfreq(48)
    radius = (fy[:, None].square() + fx[None, :].square()).sqrt()
    torch.testing.assert_close(after[..., radius <= 0.20], before[..., radius <= 0.20], rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(after[..., radius >= 0.45], before[..., radius >= 0.45] * 1.12, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(actual.mean((-2, -1)), source.mean((-2, -1)), rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(source, original, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(2, 4, 1, 1), (2, 128, 5, 7), (2, 16, 3, 7, 9)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_detail_boost_preserves_dtype_shape_and_identical_frames(shape, dtype):
    source = torch.randn(shape, dtype=dtype, generator=torch.Generator().manual_seed(456))
    source[1] = source[0]
    before = torch.random.get_rng_state().clone()
    result = core._boost_spatial_detail(source, 0.12)
    assert result.dtype == dtype and result.shape == source.shape
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result[0], result[1], rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)


def test_non_spatial_streams_and_zero_strength_are_exact_bypasses():
    packed = torch.randn(2, 1, 37)
    image = torch.randn(2, 16, 5, 7)
    assert core._boost_spatial_detail(packed, 0.12) is packed
    assert core._boost_spatial_detail(image, 0.0) is image


def test_late_window_and_patcher_clone_preserve_existing_callbacks(moss_model):
    _, positive, negative, _ = inputs()
    original_callback = lambda args: args["denoised"]
    moss_model.set_model_sampler_post_cfg_function(original_callback)
    original = guider(moss_model, positive, negative)
    before_options = copy.deepcopy(original.model_options)
    sigmas = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    active = core.guider_with_late_detail(original, sigmas, 0.12, 0.6)
    assert type(active) is type(original) and active is not original
    assert active.model_patcher is not moss_model
    assert active.original_conds is original.original_conds
    assert active.model_options["sampler_post_cfg_function"][0] is original_callback
    assert len(active.model_options["sampler_post_cfg_function"]) == 2
    callback = active.model_options["sampler_post_cfg_function"][-1]
    prediction = torch.randn(1, 16, 5, 7)
    for sigma in [1.0, 0.8, 0.6, 0.4, 0.0]:
        assert callback({"denoised": prediction, "sigma": torch.tensor([sigma])}) is prediction
    result = callback({"denoised": prediction, "sigma": torch.tensor([0.2])})
    torch.testing.assert_close(result, core._boost_spatial_detail(prediction, 0.12), rtol=1e-6, atol=1e-7)
    assert original.model_options == moss_model.model_options == before_options
    assert core.guider_with_late_detail(original, sigmas, 0.0) is original
    assert core.guider_with_late_detail(original, sigmas, 0.12, 1.0) is original
    assert core.guider_with_late_detail(original, torch.tensor([0.0]), 0.12) is original


@pytest.mark.parametrize("sampler_name", ["euler", "heun"])
@pytest.mark.parametrize("cfg", [1.0, 2.5])
def test_native_moss_sampling_preserves_early_trajectory_masks_and_call_count(moss_model, sampler_name, cfg):
    latent, positive, negative, _ = inputs()
    recorded = []

    def observe(args):
        recorded.append((float(args["sigma"][0]), args["input"].clone()))
        return args["denoised"]

    moss_model.set_model_sampler_post_cfg_function(observe)
    original = guider(moss_model, positive, negative)
    original.set_cfg(cfg)
    options_before = copy.deepcopy(original.model_options)
    sigmas = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.0])
    sigmas_before = sigmas.clone()
    node = OnekoUniversalNoiseSamplerAdvanced()
    sampler = comfy.samplers.sampler_object(sampler_name)

    def sample(**kwargs):
        return node.sample(original, sampler, sigmas, latent, True, 123, "auto",
                           "gaussian_detail", preview_method="none", **kwargs)

    baseline = sample()
    baseline_model_calls = len(moss_model.model.diffusion_model.calls)
    baseline_trace = recorded[:]
    recorded.clear()
    active = sample(detail_boost=0.12, detail_start=0.6)
    assert len(moss_model.model.diffusion_model.calls) == 2 * baseline_model_calls
    active_trace = recorded[:]
    assert len(active_trace) == len(baseline_trace)
    for (sigma_a, input_a), (sigma_b, input_b) in zip(active_trace, baseline_trace):
        assert sigma_a == sigma_b
        if sigma_a >= float(sigmas[4]):
            torch.testing.assert_close(input_a, input_b, rtol=0, atol=0)
    assert not torch.equal(active[0]["samples"], baseline[0]["samples"])
    assert torch.isfinite(active[0]["samples"]).all()
    torch.testing.assert_close(active[0]["samples"][..., :4], latent["samples"][..., :4], rtol=1e-5, atol=1e-6)
    assert active[0]["noise_mask"] is latent["noise_mask"]
    assert active[0]["batch_index"] == latent["batch_index"]
    assert original.model_options == moss_model.model_options == options_before
    torch.testing.assert_close(sigmas, sigmas_before, rtol=0, atol=0)
    repeated = sample(detail_boost=0.0)
    for output, reference in zip(repeated[:2], baseline[:2]):
        torch.testing.assert_close(output["samples"], reference["samples"], rtol=0, atol=0)


def test_failed_sampling_does_not_leave_detail_hook_on_input_guider(moss_model, monkeypatch):
    latent, positive, negative, _ = inputs()
    original = guider(moss_model, positive, negative)
    before_options = copy.deepcopy(original.model_options)

    def fail(*args, **kwargs):
        raise RuntimeError("prediction failed")

    monkeypatch.setattr(moss_model.model.diffusion_model, "forward", fail)
    with pytest.raises(RuntimeError, match="prediction failed"):
        OnekoUniversalNoiseSamplerAdvanced().sample(
            original, comfy.samplers.sampler_object("euler"), torch.tensor([1.0, 0.4, 0.2, 0.0]),
            latent, True, 123, "cpu", preview_method="none", detail_boost=0.12,
        )
    assert original.model_options == moss_model.model_options == before_options


def test_overlapping_detail_callbacks_keep_their_own_strength(moss_model):
    _, positive, negative, _ = inputs()
    original = guider(moss_model, positive, negative)
    before_options = copy.deepcopy(original.model_options)
    sigmas = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    strengths = [0.08, 0.15]
    clones = [core.guider_with_late_detail(original, sigmas, strength, 0.6) for strength in strengths]
    prediction = torch.randn(2, 16, 17, 23, generator=torch.Generator().manual_seed(999))
    args = {"denoised": prediction, "sigma": torch.tensor([0.2])}
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = [pool.submit(clone.model_options["sampler_post_cfg_function"][-1], args) for clone in clones]
        results = [task.result() for task in tasks]
    for result, strength in zip(results, strengths):
        torch.testing.assert_close(result, core._boost_spatial_detail(prediction, strength), rtol=1e-6, atol=1e-7)
    assert not torch.equal(results[0], results[1])
    assert original.model_options == moss_model.model_options == before_options
