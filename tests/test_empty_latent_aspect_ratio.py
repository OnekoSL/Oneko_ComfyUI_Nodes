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
