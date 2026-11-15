"""MOSS-format integration through native CFG, masking, scheduling and samplers.

Only the expensive diffusion prediction and model residency preparation are
substituted. Core MOSS conditioning, Flow/Flux2 latent transforms, CFGGuider,
mask preparation, Euler/DPM2 and callback/denoised handling remain native.
"""

import copy
from types import SimpleNamespace

import pytest
import torch

import nodes as core_nodes
import comfy.model_base
import comfy.model_patcher
import comfy.sample
import comfy.sampler_helpers
import comfy.samplers
import comfy.supported_models
import latent_preview
from comfy.patcher_extension import WrappersMP
from comfy_extras import nodes_custom_sampler as native_custom
from Oneko_ComfyUI_Nodes.nodes import noise_sampler_core as noise_core
from Oneko_ComfyUI_Nodes.nodes.noise_profile_cycler import OnekoNoiseProfileCycler
from Oneko_ComfyUI_Nodes.nodes.universal_noise_sampler import (
    OnekoUniversalKSampler,
    OnekoUniversalNoiseSampler,
    OnekoUniversalNoiseSamplerAdvanced,
)


SEED = 20261113


class RecordingPrediction(torch.nn.Module):
    dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.video_embedder = SimpleNamespace(weight=torch.empty(1, 128))
        self.calls = []

    def forward(self, x, timestep, context, transformer_options, **kwargs):
        refs = kwargs.get("ref_latents", [])
        self.calls.append({
            "shape": tuple(x.shape),
            "text_length": context.shape[1],
            "references": [value.clone() for value in refs],
            "reference_method": kwargs.get("ref_latents_method"),
        })
        context_value = context.mean(dim=(1, 2)).reshape(-1, 1, 1, 1)
        return (x * 0.125 + context_value * 0.01).tanh()


@pytest.fixture
def moss_model(monkeypatch):
    config = comfy.supported_models.MOSSImage({"disable_unet_model_creation": True})
    model = comfy.model_base.MOSSImage(config, device=torch.device("cpu"))
    model.diffusion_model = RecordingPrediction()
    patcher = comfy.model_patcher.ModelPatcher(
        model, load_device=torch.device("cpu"), offload_device=torch.device("cpu")
    )
    # Model residency is irrelevant to these weight-free CPU predictions.
    monkeypatch.setattr(comfy.sampler_helpers, "prepare_sampling",
                        lambda model, shape, conds, model_options: (model.model, conds, []))
    monkeypatch.setattr(latent_preview, "get_previewer", lambda *args, **kwargs: None)
    return patcher


def inputs(empty=False):
    samples = torch.linspace(-0.4, 0.8, 2 * 128 * 6 * 8).reshape(2, 128, 6, 8)
    if empty:
        samples.zero_()
    mask = torch.ones(1, 96, 128)
    mask[:, :, :64] = 0
    reference = torch.linspace(0.1, 0.3, 128 * 2 * 3).reshape(1, 128, 2, 3)
    latent = {
        "samples": samples,
        "noise_mask": mask,
        "batch_index": [3, 1],
        "downscale_ratio_spacial": 16,
        "oneko_test_tag": "preserve this metadata",
    }
    # References belong to CONDITIONING in Core, not arbitrary latent keys.
    metadata = {"reference_latents": [reference], "reference_latents_method": "index"}
    positive = [[torch.full((1, 7, 16), 0.75), metadata.copy()]]
    negative = [[torch.full((1, 3, 16), -0.25), metadata.copy()]]
    return latent, positive, negative, reference


def guider(model, positive, negative):
    result = comfy.samplers.CFGGuider(model)
    result.set_conds(positive, negative)
    result.set_cfg(2.5)
    return result


def observe_sampling(model):
    observed = []

    def observe(executor, noise, latent, sampler, sigmas, denoise_mask, callback,
                disable_pbar, seed, **kwargs):
        observed.append({"noise": noise.clone(), "latent": latent.clone(),
                         "mask": denoise_mask.clone(), "sigmas": sigmas.clone(), "seed": seed})
        return executor(noise, latent, sampler, sigmas, denoise_mask, callback,
                        disable_pbar, seed, **kwargs)

    model.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, "oneko_moss_test", observe)
    return observed


@pytest.mark.parametrize("profile", noise_core.UNIVERSAL_NOISE_PROFILES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_moss_all_profiles_are_finite_deterministic_and_keep_batch_identity(profile, dtype):
    latent = {"samples": torch.ones(3, 128, 4, 6, dtype=dtype), "batch_index": [2, 0, 2]}
    before = torch.random.get_rng_state().clone()
    generator = noise_core.make_noise_generator(SEED, "auto", profile, 0.7, 0.35)
    result = generator.generate_noise(latent)
    repeated = generator.generate_noise(latent)
    assert result.shape == latent["samples"].shape and result.dtype == dtype
    assert result.device.type == "cpu" and torch.isfinite(result).all()
    assert result.abs().sum() > 0
    torch.testing.assert_close(result, repeated, rtol=0, atol=0)
    torch.testing.assert_close(result[0], result[2], rtol=0, atol=0)
    assert not torch.equal(result[0], result[1])
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    torch.testing.assert_close(latent["samples"], torch.ones_like(latent["samples"]), rtol=0, atol=0)


@pytest.mark.parametrize("channels", [4, 16, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_index", [None, [2, 0, 2]])
def test_gaussian_matches_core_for_legacy_and_moss_formats(channels, dtype, batch_index):
    latent = {"samples": torch.ones(3, channels, 5, 7, dtype=dtype)}
    if batch_index is not None:
        latent["batch_index"] = batch_index
    actual = noise_core.make_noise_generator(SEED, "auto", "gaussian", 1.0).generate_noise(latent)
    with torch.random.fork_rng(devices=[]):
        expected = comfy.sample.prepare_noise(latent["samples"], SEED, batch_index)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cycler_drives_every_moss_noise_profile_and_wraps():
    cycler = OnekoNoiseProfileCycler()
    latent = {"samples": torch.zeros(1, 128, 4, 6)}
    profiles = []
    for index in range(len(noise_core.UNIVERSAL_NOISE_PROFILES)):
        profile, label, chosen, raw, count = cycler.select(index, "all")
        assert (label, chosen, raw, count) == (profile, index, index, len(noise_core.UNIVERSAL_NOISE_PROFILES))
        noise = noise_core.make_noise_generator(SEED, "cpu", profile).generate_noise(latent)
        assert noise.shape == latent["samples"].shape and torch.isfinite(noise).all()
        profiles.append(profile)
    assert profiles == noise_core.UNIVERSAL_NOISE_PROFILES
    assert cycler.select(len(profiles), "all")[0] == profiles[0]
    assert [cycler.select(i, "basic", 2, 4)[0] for i in range(4)] == ["laplacian", "pink", "brown", "laplacian"]


@pytest.mark.parametrize("sampler_name", ["euler", "dpm_2"])
@pytest.mark.parametrize("denoise", [1.0, 0.5])
def test_moss_universal_ksampler_gaussian_matches_native_ksampler(moss_model, sampler_name, denoise):
    latent, positive, negative, reference = inputs()
    observed = observe_sampling(moss_model)
    actual, denoised, seed = OnekoUniversalKSampler().sample(
        moss_model, positive, negative, latent, SEED, 4, 2.5, sampler_name,
        "simple", denoise, True, "auto", preview_method="none",
    )
    first = observed[-1]
    with torch.random.fork_rng(devices=[]):
        expected = core_nodes.common_ksampler(
            moss_model, SEED, 4, 2.5, sampler_name, "simple", positive, negative, latent,
            denoise=denoise,
        )[0]
    native = observed[-1]
    assert seed == SEED and first["seed"] == SEED
    torch.testing.assert_close(first["noise"], native["noise"], rtol=0, atol=0)
    torch.testing.assert_close(first["sigmas"], native["sigmas"], rtol=0, atol=0)
    torch.testing.assert_close(actual["samples"], expected["samples"], rtol=0, atol=0)
    assert torch.isfinite(denoised["samples"]).all()


@pytest.mark.parametrize("node_kind", ["ksampler", "custom", "advanced"])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("add_noise", [False, True])
def test_moss_samplers_preserve_mask_references_metadata_and_native_text_lengths(
    moss_model, node_kind, empty, add_noise,
):
    latent, positive, negative, reference = inputs(empty)
    original = copy.deepcopy(latent)
    reference_before = reference.clone()
    observed = observe_sampling(moss_model)
    options_before = copy.deepcopy(moss_model.model_options)
    sigma_values = torch.tensor([1.0, 0.8, 0.5, 0.2, 0.0])
    sigmas_before = sigma_values.clone()
    if node_kind == "ksampler":
        actual, denoised, seed = OnekoUniversalKSampler().sample(
            moss_model, positive, negative, latent, SEED, 4, 2.5, "euler",
            "simple", 1.0, add_noise, "auto", preview_method="none",
        )
    else:
        cls = OnekoUniversalNoiseSampler if node_kind == "custom" else OnekoUniversalNoiseSamplerAdvanced
        actual, denoised, seed = cls().sample(
            guider(moss_model, positive, negative), comfy.samplers.sampler_object("euler"),
            sigma_values, latent, add_noise, SEED, "auto", preview_method="none",
        )
    assert actual["samples"].shape == (2, 128, 6, 8)
    assert seed == observed[-1]["seed"] == SEED
    assert observed[-1]["mask"].shape == (2, 128, 6, 8)
    assert (observed[-1]["mask"][..., :4] == 0).all()
    assert (observed[-1]["mask"][..., 4:] == 1).all()
    if not add_noise:
        assert torch.count_nonzero(observed[-1]["noise"]) == 0
    torch.testing.assert_close(actual["samples"][..., :4], original["samples"][..., :4], rtol=1e-5, atol=1e-6)
    assert actual["oneko_test_tag"] == original["oneko_test_tag"]
    assert "downscale_ratio_spacial" not in actual
    assert actual["noise_mask"] is latent["noise_mask"]
    assert actual["batch_index"] == [3, 1]
    assert torch.isfinite(actual["samples"]).all() and torch.isfinite(denoised["samples"]).all()
    assert {call["text_length"] for call in moss_model.model.diffusion_model.calls} == {3, 7}
    expected_ref = moss_model.model.process_latent_in(reference).repeat(2, 1, 1, 1)
    for call in moss_model.model.diffusion_model.calls:
        assert call["shape"] == (2, 128, 6, 8)
        assert call["reference_method"] == "index"
        assert len(call["references"]) == 1
        torch.testing.assert_close(call["references"][0], expected_ref, rtol=0, atol=0)
    torch.testing.assert_close(latent["samples"], original["samples"], rtol=0, atol=0)
    torch.testing.assert_close(latent["noise_mask"], original["noise_mask"], rtol=0, atol=0)
    torch.testing.assert_close(reference, reference_before, rtol=0, atol=0)
    torch.testing.assert_close(sigma_values, sigmas_before, rtol=0, atol=0)
    assert moss_model.model_options == options_before


@pytest.mark.parametrize("leftover,expected_sigmas", [
    ("disable", [0.8, 0.5, 0.0]), ("enable", [0.8, 0.5, 0.2]),
])
def test_moss_advanced_step_range_matches_native_custom_sampler(moss_model, leftover, expected_sigmas):
    latent, positive, negative, _ = inputs()
    sigmas = torch.tensor([1.0, 0.8, 0.5, 0.2, 0.0])
    actual = OnekoUniversalNoiseSamplerAdvanced().sample(
        guider(moss_model, positive, negative), comfy.samplers.sampler_object("euler"),
        sigmas, latent, True, SEED, "auto", start_at_step=1, end_at_step=3,
        return_with_leftover_noise=leftover, preview_method="none",
    )
    with torch.random.fork_rng(devices=[]):
        expected = native_custom.SamplerCustomAdvanced.execute(
            native_custom.Noise_RandomNoise(SEED), guider(moss_model, positive, negative),
            comfy.samplers.sampler_object("euler"), torch.tensor(expected_sigmas), latent,
        ).result
    for actual_latent, native_latent in zip(actual[:2], expected):
        torch.testing.assert_close(actual_latent["samples"], native_latent["samples"], rtol=0, atol=0)
        assert actual_latent.keys() == native_latent.keys()
    torch.testing.assert_close(sigmas, torch.tensor([1.0, 0.8, 0.5, 0.2, 0.0]), rtol=0, atol=0)


def test_moss_advanced_empty_step_range_returns_original_samples_without_prediction(moss_model):
    latent, positive, negative, _ = inputs()
    actual, denoised, seed = OnekoUniversalNoiseSamplerAdvanced().sample(
        guider(moss_model, positive, negative), comfy.samplers.sampler_object("euler"),
        torch.tensor([1.0, 0.5, 0.0]), latent, True, SEED, "auto",
        start_at_step=2, preview_method="none",
    )
    assert seed == SEED and actual is denoised
    assert actual["samples"] is latent["samples"]
    assert actual["noise_mask"] is latent["noise_mask"]
    assert "downscale_ratio_spacial" not in actual
    assert moss_model.model.diffusion_model.calls == []
