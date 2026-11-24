"""Controlled noise must retain native endpoints, scale and batch identity."""

import pytest
import torch

import comfy.nested_tensor
import comfy.sample
import comfy.samplers
from comfy_extras import nodes_custom_sampler as native_custom
from Oneko_ComfyUI_Nodes.nodes import noise_sampler_core as core
from Oneko_ComfyUI_Nodes.nodes.universal_noise_sampler import OnekoUniversalNoiseSamplerAdvanced
from .test_moss_sampling_integration import guider, inputs, moss_model


SEED = 20261115


@pytest.mark.parametrize("profile", core.UNIVERSAL_NOISE_PROFILES)
def test_zero_profile_blend_matches_native_gaussian(profile):
    latent = {"samples": torch.zeros(3, 128, 5, 7), "batch_index": [2, 0, 2]}
    before = torch.random.get_rng_state().clone()
    actual = core.make_noise_generator(SEED, "auto", profile, profile_blend=0).generate_noise(latent)
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    with torch.random.fork_rng(devices=[]):
        native = comfy.sample.prepare_noise(latent["samples"], SEED, latent["batch_index"])
    torch.testing.assert_close(actual, native, rtol=0, atol=0)


@pytest.mark.parametrize("profile", core.UNIVERSAL_NOISE_PROFILES)
def test_blending_preserves_each_samples_scale_and_identity(profile):
    latent = {"samples": torch.zeros(3, 128, 5, 7), "batch_index": [2, 0, 2]}
    base = core.make_noise_generator(SEED, "auto").generate_noise(latent)
    generator = core.make_noise_generator(SEED, "auto", profile, profile_blend=0.2)
    before = torch.random.get_rng_state().clone()
    actual = generator.generate_noise(latent)
    repeated = generator.generate_noise(latent)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
    torch.testing.assert_close(actual[0], actual[2], rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    dims = (1, 2, 3)
    torch.testing.assert_close(actual.mean(dims), base.mean(dims), rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(actual.std(dims), base.std(dims), rtol=1e-6, atol=1e-6)
    assert torch.count_nonzero(latent["samples"]) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("profile", ["gaussian_detail", "pyramid_mix", "violet", "illustrious_texture"])
def test_blends_keep_dtype_and_apply_strength_once(dtype, profile):
    latent = {"samples": torch.zeros(2, 16, 7, 9, dtype=dtype)}
    full = core.make_noise_generator(SEED, "cpu", profile, profile_blend=0.2).generate_noise(latent)
    scaled = core.make_noise_generator(SEED, "cpu", profile, 0.5, profile_blend=0.2).generate_noise(latent)
    zero = core.make_noise_generator(SEED, "cpu", profile, 0, profile_blend=0.2).generate_noise(latent)
    assert full.dtype == scaled.dtype == zero.dtype == dtype
    torch.testing.assert_close(scaled, full * 0.5, rtol=0, atol=0)
    assert torch.isfinite(full).all() and torch.count_nonzero(zero) == 0


@pytest.mark.parametrize("shape", [(2, 4, 1, 1), (2, 16, 7, 9), (2, 128, 3, 5), (2, 8, 3, 7, 9)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_detail_zero_is_exactly_gaussian_for_image_and_video_formats(shape, dtype):
    latent = {"samples": torch.zeros(shape, dtype=dtype)}
    actual = core.make_noise_generator(SEED, "auto", "gaussian_detail", detail_bias=0).generate_noise(latent)
    with torch.random.fork_rng(devices=[]):
        native = comfy.sample.prepare_noise(latent["samples"], SEED)
    torch.testing.assert_close(actual, native, rtol=0, atol=0)
    active = core.make_noise_generator(SEED, "auto", "gaussian_detail", detail_bias=1).generate_noise(latent)
    assert active.shape == native.shape and active.dtype == dtype
    assert torch.isfinite(active).all()


def test_detail_control_increases_spatial_high_frequencies_without_randomizing_phases():
    latent = {"samples": torch.zeros(2, 16, 32, 48)}
    base = core.make_noise_generator(SEED, "cpu").generate_noise(latent)
    base_fft = torch.fft.rfft2(base)
    fy = torch.fft.fftfreq(32)
    fx = torch.fft.rfftfreq(48)
    radius_squared = fy[:, None].square() + fx[None, :].square()
    low = (radius_squared > 0) & (radius_squared < 0.03)
    high = radius_squared > 0.2
    ratios = []
    for bias in [0.0, 0.35, 1.0]:
        actual = core.make_noise_generator(SEED, "cpu", "gaussian_detail", detail_bias=bias).generate_noise(latent)
        spectrum = torch.fft.rfft2(actual)
        power = spectrum.abs().square()
        ratios.append((power[..., high].mean() / power[..., low].mean()).item())
        phase = (spectrum * base_fft.conj())[..., radius_squared > 0]
        assert phase.real.min() > 0
        unit_phase = phase / phase.abs()
        torch.testing.assert_close(unit_phase.imag, torch.zeros_like(unit_phase.imag), rtol=0, atol=1e-4)
        torch.testing.assert_close(actual.std((1, 2, 3)), base.std((1, 2, 3)), rtol=1e-6, atol=1e-6)
        correlation = torch.corrcoef(torch.stack([actual.flatten(), base.flatten()]))[0, 1]
        assert correlation > 0.98
    assert ratios[0] < ratios[1] < ratios[2]


@pytest.mark.parametrize("profile,blend", [("gaussian_detail", 1.0), ("gaussian_detail", 0.2), ("pyramid_mix", 0.2)])
def test_detail_and_blending_preserve_nested_streams(profile, blend):
    streams = [torch.zeros(3, 16, 5, 7), torch.zeros(3, 4, 11)]
    latent = {"samples": comfy.nested_tensor.NestedTensor(streams), "batch_index": [1, 0, 1]}
    base = core.make_noise_generator(SEED, "cpu").generate_noise(latent)
    result = core.make_noise_generator(SEED, "cpu", profile, profile_blend=blend).generate_noise(latent)
    assert result.is_nested
    for actual, original in zip(result.unbind(), base.unbind()):
        assert actual.shape == original.shape and torch.isfinite(actual).all()
        torch.testing.assert_close(actual[0], actual[2], rtol=0, atol=0)
        dims = tuple(range(1, actual.ndim))
        torch.testing.assert_close(actual.std(dims), original.std(dims), rtol=1e-6, atol=1e-6)


def test_advanced_profile_blend_is_optional_and_keeps_legacy_default():
    schema = OnekoUniversalNoiseSamplerAdvanced.INPUT_TYPES()
    assert schema["optional"]["profile_blend"][1]["default"] == 1.0
    assert list(schema["required"])[-1] == "preview_method"


@pytest.mark.parametrize("profile,blend,add_noise", [
    ("pyramid_mix", 0.2, True), ("gaussian_detail", 1.0, True),
    ("gaussian_detail", 0.0, True), ("gaussian_detail", 0.2, False),
])
def test_advanced_blend_matches_native_sampling_with_the_same_noise(moss_model, profile, blend, add_noise):
    latent, positive, negative, _ = inputs()
    sigmas = torch.tensor([1.0, 0.7, 0.4, 0.0])
    sampler = comfy.samplers.sampler_object("euler")
    actual = OnekoUniversalNoiseSamplerAdvanced().sample(
        guider(moss_model, positive, negative), sampler, sigmas, latent,
        add_noise, SEED, "auto", profile, detail_bias=0.35,
        preview_method="none", profile_blend=blend,
    )
    noise = core.make_noise_generator(SEED, "auto", profile, detail_bias=0.35, profile_blend=blend)
    if not add_noise:
        noise = core.OnekoEmptyNoise()
        noise.seed = SEED
    expected = native_custom.SamplerCustomAdvanced.execute(
        noise, guider(moss_model, positive, negative), sampler, sigmas, latent,
    ).result
    assert actual[2] == SEED
    for output, reference in zip(actual[:2], expected):
        torch.testing.assert_close(output["samples"], reference["samples"], rtol=0, atol=0)
