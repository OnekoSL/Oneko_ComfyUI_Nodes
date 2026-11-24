from types import SimpleNamespace

import pytest
import torch

import comfy.latent_formats
import comfy.sample
from comfy_extras.nodes_flux import EmptyFlux2LatentImage
from comfy_extras.nodes_resolution import AspectRatio

from Oneko_ComfyUI_Nodes.nodes.empty_latent_aspect_ratio import OnekoEmptyLatentAspectRatio


@pytest.mark.parametrize("latent_format,channels,divisor", [
    ("auto", 4, 8),
    ("SD1 / SDXL", 4, 8),
    ("SD3 / FLUX.1", 16, 8),
    ("MOSS / FLUX.2", 128, 16),
])
def test_custom_dimensions_batch_and_empty_values(latent_format, channels, divisor):
    latent, width, height = OnekoEmptyLatentAspectRatio.execute(
        aspect_ratio="custom", width=256, height=384, latent_format=latent_format, batch_size=2,
    ).result
    assert (width, height) == (256, 384)
    assert latent["samples"].shape == (2, channels, 384 // divisor, 256 // divisor)
    assert torch.count_nonzero(latent["samples"]) == 0
    assert latent["downscale_ratio_spacial"] == divisor


@pytest.mark.parametrize("source_format", ["auto", "SD1 / SDXL", "SD3 / FLUX.1", "MOSS / FLUX.2"])
@pytest.mark.parametrize("target_format", [comfy.latent_formats.SDXL, comfy.latent_formats.SD3, comfy.latent_formats.Flux2])
def test_sampler_adaptation_keeps_pixel_dimensions(source_format, target_format):
    latent, width, height = OnekoEmptyLatentAspectRatio.execute(
        aspect_ratio="custom", width=256, height=384, latent_format=source_format,
    ).result
    target = target_format()
    model = SimpleNamespace(get_model_object=lambda name: target)
    samples = comfy.sample.fix_empty_latent_channels(model, latent["samples"], latent["downscale_ratio_spacial"])
    assert samples.shape == (1, target.latent_channels, height // target.spacial_downscale_ratio, width // target.spacial_downscale_ratio)
    assert samples.shape[-1] * target.spacial_downscale_ratio == width
    assert samples.shape[-2] * target.spacial_downscale_ratio == height


def test_moss_portrait_and_swapped_landscape_do_not_double_dimensions():
    for swap, dimensions, latent_hw in [(False, (1024, 1536), (96, 64)), (True, (1536, 1024), (64, 96))]:
        latent, width, height = OnekoEmptyLatentAspectRatio.execute(
            megapixels=1.5, swap_dimensions=swap, latent_format="MOSS / FLUX.2",
        ).result
        assert (width, height) == dimensions
        assert latent["samples"].shape[-2:] == latent_hw
        native = EmptyFlux2LatentImage.execute(width, height).result[0]
        torch.testing.assert_close(latent["samples"], native["samples"], rtol=0, atol=0)


def test_aspect_preset_uses_megapixels_instead_of_custom_inputs():
    _, width, height = OnekoEmptyLatentAspectRatio.execute(
        aspect_ratio=AspectRatio.WIDESCREEN_H.value, megapixels=1.0, width=512, height=512,
    ).result
    assert (width, height) == (1360, 768)


@pytest.mark.parametrize("latent_format,channels,divisor", [
    ("SD1 / SDXL", 4, 8),
    ("MOSS / FLUX.2", 128, 16),
])
@pytest.mark.parametrize("aspect_ratio,dimensions8,dimensions16", [
    ("5:4 (Classic Monitor)", (1400, 1120), (1408, 1120)),
    ("16:10 (Desktop Wallpaper)", (1584, 992), (1584, 992)),
    ("32:9 (Super Ultrawide)", (2368, 664), (2368, 672)),
    ("32:10 (Dual Monitor)", (2240, 704), (2240, 704)),
    ("9:20 (Phone Wallpaper)", (840, 1872), (848, 1872)),
    ("2:1 (Banner)", (1776, 888), (1776, 880)),
    ("3:1 (Wide Banner)", (2176, 728), (2176, 720)),
    ("4:1 (Panoramic Banner)", (2512, 624), (2512, 624)),
    ("5:1 (Header Banner)", (2808, 560), (2800, 560)),
])
def test_banner_and_wallpaper_dimensions(aspect_ratio, dimensions8, dimensions16, latent_format, channels, divisor):
    latent, width, height = OnekoEmptyLatentAspectRatio.execute(
        aspect_ratio=aspect_ratio, megapixels=1.5, width=512, height=512, latent_format=latent_format,
    ).result
    assert (width, height) == (dimensions16 if divisor == 16 else dimensions8)
    assert latent["samples"].shape == (1, channels, height // divisor, width // divisor)
    assert latent["downscale_ratio_spacial"] == divisor


def test_banner_megapixels_swap_and_scale():
    latent, width, height = OnekoEmptyLatentAspectRatio.execute(
        aspect_ratio="4:1 (Panoramic Banner)", megapixels=1.0,
        swap_dimensions=True, scale_factor=0.5, latent_format="MOSS / FLUX.2",
    ).result
    assert (width, height) == (256, 1024)
    assert latent["samples"].shape == (1, 128, 64, 16)


def test_swap_then_scale_and_return_final_dimensions():
    latent, width, height = OnekoEmptyLatentAspectRatio.execute(
        swap_dimensions=True, scale_factor=0.5, latent_format="MOSS / FLUX.2",
    ).result
    assert (width, height) == (768, 512)
    assert latent["samples"].shape == (1, 128, 32, 48)


@pytest.mark.parametrize("latent_format,expected", [("auto", (1024, 1536)), ("MOSS / FLUX.2", (1024, 1536)), ("SD1 / SDXL", (1032, 1544))])
def test_custom_dimensions_are_aligned_before_creating_latent(latent_format, expected):
    _, width, height = OnekoEmptyLatentAspectRatio.execute(
        aspect_ratio="custom", width=1030, height=1542, latent_format=latent_format,
    ).result
    assert (width, height) == expected


def test_small_scaled_dimensions_remain_nonempty():
    latent, width, height = OnekoEmptyLatentAspectRatio.execute(
        aspect_ratio="custom", width=16, height=16, scale_factor=0.125, latent_format="MOSS / FLUX.2",
    ).result
    assert (width, height) == (16, 16)
    assert latent["samples"].shape == (1, 128, 1, 1)


def test_empty_latent_creation_does_not_advance_random_seed():
    before = torch.random.get_rng_state().clone()
    OnekoEmptyLatentAspectRatio.execute(aspect_ratio="custom", width=64, height=64)
    assert torch.equal(torch.random.get_rng_state(), before)


def test_unknown_format_is_rejected():
    with pytest.raises(ValueError, match="Unknown latent format"):
        OnekoEmptyLatentAspectRatio.execute(latent_format="unknown")
