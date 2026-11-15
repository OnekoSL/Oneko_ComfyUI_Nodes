"""MOSS HiResFix checks against the installed USDU tile pipeline, without weights."""

import copy
import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import comfy.latent_formats
import comfy.model_patcher
import comfy.sd
from Oneko_ComfyUI_Nodes.nodes import hiresfix_tiled as hires
from tests.paths import COMFY_ROOT


@pytest.fixture(scope="module")
def native_usdu():
    directory = Path(COMFY_ROOT) / "custom_nodes" / "ComfyUI_UltimateSDUpscale"
    init = directory / "__init__.py"
    bundled_script = directory / "repositories" / "ultimate_sd_upscale" / "scripts" / "ultimate-upscale.py"
    if not init.is_file() or not bundled_script.is_file():
        pytest.skip("Installed USDU with its bundled script is required; tests never download it")
    name = "_oneko_hires_native_usdu"
    spec = importlib.util.spec_from_file_location(name, init, submodule_search_locations=[str(directory)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return package.NODE_CLASS_MAPPINGS["UltimateSDUpscaleNoUpscale"]


class SmallVAE:
    crop_input = True
    output_channels = 3
    pad_channel_value = None

    def __init__(self, compression):
        self.compression = compression
        self.channels = 128 if compression == 16 else 4
        self.encodes = []
        self.tiled_decodes = []

    def spacial_compression_encode(self):
        return self.compression

    def spacial_compression_decode(self):
        return self.compression

    def temporal_compression_decode(self):
        return None

    def encode(self, pixels):
        # Native VAE alignment, small deterministic stand-in only for the weights.
        cropped = comfy.sd.VAE.vae_encode_crop_pixels(self, pixels)
        samples = F.avg_pool2d(cropped.movedim(-1, 1), self.compression).mean(1, keepdim=True)
        samples = samples.repeat(1, self.channels, 1, 1)
        self.encodes.append((tuple(pixels.shape), tuple(cropped.shape), samples.clone()))
        return samples

    def decode(self, samples):
        return F.interpolate(samples[:, :3], scale_factor=self.compression, mode="nearest-exact").movedim(1, -1)

    def decode_tiled(self, samples, **kwargs):
        self.tiled_decodes.append(kwargs)
        return self.decode(samples)


def make_model(compression=16):
    host = torch.nn.Module()
    host.latent_format = comfy.latent_formats.Flux2() if compression == 16 else comfy.latent_formats.SD15()
    host.model_sampling = SimpleNamespace(sigma_min=0.0, timestep=lambda value: value)
    return comfy.model_patcher.ModelPatcher(host, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))


def settings(image, model, vae):
    return dict(image=image, upscale_model=object(), model=model,
                positive=[[torch.ones(1, 5, 8), {"marker": "positive"}]],
                negative=[[torch.ones(1, 2, 8), {"marker": "negative"}]], vae=vae,
                seed=17, steps=2, cfg=2.5, sampler_name="euler", scheduler="simple", denoise=0.4,
                tile_width=72, tile_height=88, mask_blur=4, tile_padding=0,
                use_reference_latent=True, use_differential_diffusion=True,
                differential_strength=0.7, tiled_decode=True, batch_size=1)


def install_native_pipeline(monkeypatch, native_usdu, fail=False):
    monkeypatch.setitem(hires.nodes.NODE_CLASS_MAPPINGS, "UltimateSDUpscaleNoUpscale", native_usdu)
    processing = hires._get_usdu_processing_globals(native_usdu)
    shared = processing["shared"]
    # USDU normally owns these process-local values; restore them after each test.
    for key in ("batch", "batch_as_tensor", "actual_upscaler", "sd_upscalers"):
        if hasattr(shared, key):
            value = getattr(shared, key)
            monkeypatch.setattr(shared, key, value.copy() if isinstance(value, list) else value)
    monkeypatch.setattr(shared.state, "interrupted", False)
    monkeypatch.setattr(hires.comfy.utils, "PROGRESS_BAR_ENABLED", False)
    monkeypatch.setattr(hires.ImageUpscaleWithModel, "execute",
                        lambda model, image: (F.interpolate(image.movedim(-1, 1), scale_factor=2,
                                                          mode="nearest-exact").movedim(1, -1),))
    monkeypatch.setattr(hires.latent_preview, "prepare_callback", lambda *args: None)
    calls = []

    def sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent, **kwargs):
        calls.append(dict(model=model, noise=noise.clone(), latent=latent.clone(),
                          positive=positive, negative=negative, **kwargs))
        if fail:
            raise RuntimeError("deliberate tile failure")
        return latent + 0.01

    monkeypatch.setattr(hires.comfy.sample, "sample", sample)
    return calls, processing


@pytest.mark.parametrize("tile_batch", [1, 2])
@pytest.mark.parametrize("tiled_decode", [False, True])
@pytest.mark.parametrize("tile_padding", [0, 8])
def test_moss_native_tiles_keep_16_alignment_reference_batch_and_edges(
        monkeypatch, native_usdu, caplog, tile_batch, tiled_decode, tile_padding):
    calls, processing = install_native_pipeline(monkeypatch, native_usdu)
    previous_sample = processing["sample"]
    image = torch.stack((torch.linspace(.2, .4, 48 * 56 * 3).reshape(48, 56, 3),
                         torch.linspace(.6, .8, 48 * 56 * 3).reshape(48, 56, 3)))
    original_image = image.clone()
    vae = SmallVAE(16)
    model = make_model()
    options_before = copy.deepcopy(model.model_options)
    values = settings(image, model, vae)
    values.update(batch_size=tile_batch, tiled_decode=tiled_decode, tile_padding=tile_padding)
    positive_before = copy.deepcopy(values["positive"][0][1])
    with caplog.at_level(logging.WARNING):
        refined, upscaled, seed = hires.OnekoHiResFixTiled().hiresfix(**values)

    assert seed == 17 and refined.shape == upscaled.shape == (2, 96, 112, 3)
    assert calls and processing["sample"] is previous_sample
    assert model.model_options == options_before
    assert values["positive"][0][1] == positive_before
    torch.testing.assert_close(image, original_image, rtol=0, atol=0)
    assert torch.isfinite(refined).all() and refined.min() > 0
    # Every tile arrives aligned before the VAE's native crop, including edge tiles.
    for before, after, samples in vae.encodes:
        assert before == after
        assert samples.shape[1] == 128 and torch.count_nonzero(samples) > 0
    full_reference = vae.encodes[0][2]
    assert full_reference.shape == (2, 128, 6, 7)
    for call in calls:
        assert call["latent"].shape[1:] == (128, 6, 5)
        assert call["noise"].shape == call["latent"].shape
        assert call["noise_mask"] is None
        assert "denoise_mask_function" in call["model"].model_options
        tile_count = call["latent"].shape[0] // image.shape[0]
        refs = call["positive"][0][1]["reference_latents"]
        assert len(refs) == 1
        torch.testing.assert_close(refs[0], full_reference.repeat_interleave(tile_count, dim=0), rtol=0, atol=0)
        assert "reference_latents" not in call["negative"][0][1]
    warnings = [record.message for record in caplog.records if "supplied no noise_mask" in record.message]
    assert len(warnings) == 1  # Installed USDU uses masks for final compositing, not sampler masking.
    if tiled_decode:
        assert vae.tiled_decodes and all(entry["tile_x"] == 32 for entry in vae.tiled_decodes)
    else:
        assert not vae.tiled_decodes


@pytest.mark.parametrize("use_reference", [False, True])
def test_factor8_preserves_legacy_tile_and_reference_crop(monkeypatch, native_usdu, use_reference):
    calls, _ = install_native_pipeline(monkeypatch, native_usdu)
    image = torch.full((1, 48, 56, 3), .5)
    vae = SmallVAE(8)
    values = settings(image, make_model(8), vae)
    values.update(use_reference_latent=use_reference, use_differential_diffusion=False)
    refined, upscaled, _ = hires.OnekoHiResFixTiled().hiresfix(**values)
    assert refined.shape == upscaled.shape == (1, 96, 112, 3)
    assert calls
    for call in calls:
        assert call["latent"].shape[1:] == (4, 11, 9)
        assert "denoise_mask_function" not in call["model"].model_options
        refs = call["positive"][0][1].get("reference_latents")
        if use_reference:
            assert refs[0].shape == (1, 4, 11, 9)
        else:
            assert refs is None


@pytest.mark.parametrize("side", ["positive", "negative"])
def test_factor16_external_references_fail_before_upscale(monkeypatch, side):
    image = torch.ones(1, 32, 32, 3)
    values = settings(image, make_model(), SmallVAE(16))
    reference = torch.ones(1, 128, 2, 2)
    values[side][0][1]["reference_latents"] = [reference]
    monkeypatch.setattr(hires.ImageUpscaleWithModel, "execute", lambda *args: pytest.fail("must fail before upscaling"))
    with pytest.raises(ValueError, match="external reference_latents.*factor-16"):
        hires.OnekoHiResFixTiled().hiresfix(**values)
    assert values[side][0][1]["reference_latents"][0] is reference


def test_native_usdu_failure_restores_sample_and_caller_state(monkeypatch, native_usdu):
    _, processing = install_native_pipeline(monkeypatch, native_usdu, fail=True)
    previous_sample = processing["sample"]
    image = torch.full((1, 48, 56, 3), .5)
    model = make_model()
    values = settings(image, model, SmallVAE(16))
    before = copy.deepcopy(model.model_options)
    with pytest.raises(RuntimeError, match="deliberate tile failure"):
        hires.OnekoHiResFixTiled().hiresfix(**values)
    assert processing["sample"] is previous_sample
    assert model.model_options == before
    assert "reference_latents" not in values["positive"][0][1]


def test_supplied_noise_mask_reaches_native_differential_diffusion(monkeypatch, caplog):
    model = make_model()
    patched = hires._first_output(hires.DifferentialDiffusion.execute(model, 0.7))
    latent = {"samples": torch.ones(1, 128, 4, 5),
              "noise_mask": torch.linspace(0, 1, 20).reshape(1, 4, 5)}
    mask = latent["noise_mask"]
    monkeypatch.setattr(hires.latent_preview, "prepare_callback", lambda *args: None)

    def sample(work_model, noise, steps, cfg, sampler, scheduler, positive, negative, samples, **kwargs):
        assert kwargs["noise_mask"] is mask
        callback = work_model.model_options["denoise_mask_function"]
        result = callback(torch.tensor([.5]), mask, {
            "model": SimpleNamespace(inner_model=SimpleNamespace(model_sampling=model.model.model_sampling)),
            "sigmas": torch.tensor([1., 0.]),
        })
        expected = .7 * (mask >= .5).to(mask.dtype) + .3 * mask
        torch.testing.assert_close(result, expected)
        return samples

    monkeypatch.setattr(hires.comfy.sample, "sample", sample)
    with caplog.at_level(logging.WARNING):
        result = hires._make_usdu_noise_sample("auto", "gaussian", 1, .35, warn_missing_noise_mask=True)(
            patched, 17, 2, 2.5, "euler", "simple", [], [], latent, .4, None, None)
    assert result["noise_mask"] is mask
    assert not any("supplied no noise_mask" in record.message for record in caplog.records)
