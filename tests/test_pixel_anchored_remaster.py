import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


from tests.paths import COMFY_ROOT, ONEKO_ROOT
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

MODULE_PATH = ONEKO_ROOT / "nodes" / "pixel_anchored_remaster.py"
spec = importlib.util.spec_from_file_location("pixel_anchored_remaster", MODULE_PATH)
remaster = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remaster)


class PixelAnchoredRemasterTests(unittest.TestCase):
    def test_scaled_size_snaps_to_multiple_of_8_and_clamps_minimum(self):
        self.assertEqual(remaster._scaled_size(704, 1248, 0.75), (528, 936))
        self.assertEqual(remaster._scaled_size(9, 9, 0.25), (8, 8))
        self.assertEqual(remaster._scaled_size(512, 512, 2.0), (512, 512))

    def test_combo_default_uses_preferred_when_available(self):
        self.assertEqual(remaster._combo_default(["a", "res_2m"], "res_2m"), "res_2m")
        self.assertEqual(remaster._combo_default(["fallback", "other"], "res_2m"), "fallback")

    def test_node_registration_and_interface(self):
        node = remaster.NODE_CLASS_MAPPINGS["OnekoPixelAnchoredRemaster"]
        inputs = node.INPUT_TYPES()["required"]

        self.assertEqual(node.CATEGORY, "Oneko/06 Image/Upscale")
        self.assertEqual(
            remaster.NODE_DISPLAY_NAME_MAPPINGS["OnekoPixelAnchoredRemaster"],
            "Pixel Anchored Remaster (Oneko)",
        )
        self.assertEqual(node.RETURN_NAMES, ("final_image", "downscaled_image", "remaster_latent", "seed", "settings_report"))
        self.assertEqual(inputs["pixel_scale"][1]["default"], 0.90)
        self.assertEqual(inputs["steps"][1]["default"], 12)
        self.assertEqual(inputs["cfg"][1]["default"], 2.5)
        self.assertEqual(inputs["denoise"][1]["default"], 0.16)
        self.assertEqual(inputs["remaster_blend"][1]["default"], 0.35)
        self.assertEqual(inputs["seed"][1]["control_after_generate"], True)
        self.assertEqual(inputs["use_reference_latent"][1]["default"], True)


    def test_default_sampler_and_scheduler_fall_back_when_preferred_is_missing(self):
        fake_sampler = type("FakeKSampler", (), {"SAMPLERS": ["sampler_a"], "SCHEDULERS": ["scheduler_a"]})
        with patch.object(remaster.comfy.samplers, "KSampler", fake_sampler):
            inputs = remaster.OnekoPixelAnchoredRemaster.INPUT_TYPES()["required"]

        self.assertEqual(inputs["sampler_name"][1]["default"], "sampler_a")
        self.assertEqual(inputs["scheduler"][1]["default"], "scheduler_a")




class _ImageVAE:
    """CPU double; the real Core tiled encode/decode nodes wrap these methods."""

    def __init__(self, compression, channels, decode_compression=None):
        self.compression = compression
        self.channels = channels
        self.decode_compression = decode_compression or compression
        self.encoded = None
        self.encode_pixels = None
        self.decode_samples = None

    def spacial_compression_encode(self):
        return self.compression

    def spacial_compression_decode(self):
        return self.decode_compression

    def temporal_compression_decode(self):
        return None

    def encode_tiled(self, pixels, **kwargs):
        self.encode_pixels = pixels
        height, width = pixels.shape[1:3]
        assert height % self.compression == width % self.compression == 0
        reduced = torch.nn.functional.interpolate(
            pixels.movedim(-1, 1)[:, :1],
            size=(height // self.compression, width // self.compression),
            mode="area",
        )
        self.encoded = reduced.expand(-1, self.channels, -1, -1).clone()
        return self.encoded

    def decode_tiled(self, samples, **kwargs):
        self.decode_samples = samples
        return torch.nn.functional.interpolate(
            samples[:, :3], scale_factor=self.decode_compression, mode="nearest"
        ).movedim(1, -1)


def _run_remaster(vae, width, height, *, batch=1, denoise=0.16, blend=1.0,
                  reference=True, method="bilinear", pixel_scale=0.5, pattern=False):
    image = (torch.arange(batch, dtype=torch.float32) * 0.1 + 0.2).reshape(batch, 1, 1, 1)
    image = image.expand(batch, height, width, 3).clone()
    if pattern:
        image += torch.linspace(0, 0.1, width).reshape(1, 1, width, 1)
        image += torch.linspace(0, 0.15, height).reshape(1, height, 1, 1)
    positive = [[torch.zeros(1, 2, 3), {"keep": "positive"}]]
    negative = [[torch.zeros(1, 2, 3), {"keep": "negative"}]]

    def sample(model, seed, steps, cfg, sampler, scheduler, pos, neg, latent, denoise):
        result = latent.copy()
        result["samples"] = latent["samples"] + 0.2
        return (result,)

    with patch.object(remaster, "common_ksampler", side_effect=sample) as sampler:
        outputs = remaster.OnekoPixelAnchoredRemaster().remaster(
            image=image, model=object(), positive=positive, negative=negative,
            vae=vae, seed=12345, pixel_scale=pixel_scale, pixel_method="area",
            latent_upscale_method=method, steps=12, cfg=2.5,
            sampler_name="euler", scheduler="simple", denoise=denoise,
            remaster_blend=blend, vae_tile_size=1024, vae_overlap=64,
            use_reference_latent=reference,
        )
    return outputs, sampler, positive, negative


@pytest.mark.parametrize(
    "compression,channels,width,height,final_width,final_height,batch",
    [
        (16, 128, 1024, 1536, 1024, 1536, 1),
        (16, 128, 1536, 1024, 1536, 1024, 1),
        (16, 128, 111, 87, 96, 80, 2),
        (8, 4, 111, 87, 104, 80, 2),
        (8, 16, 111, 87, 104, 80, 2),
        (8, 4, 9, 9, 64, 64, 1),
        (16, 128, 9, 9, 64, 64, 1),
    ],
)
def test_pipeline_uses_vae_size_and_preserves_batch(
    compression, channels, width, height, final_width, final_height, batch
):
    vae = _ImageVAE(compression, channels)
    (image, downscaled, latent, seed, report), sampler, positive, negative = _run_remaster(
        vae, width, height, batch=batch
    )

    assert image.shape == (batch, final_height, final_width, 3)
    assert latent["samples"].shape == (
        batch, channels, final_height // compression, final_width // compression
    )
    assert downscaled.shape[1] % compression == downscaled.shape[2] % compression == 0
    assert downscaled.shape[1] >= compression and downscaled.shape[2] >= compression
    assert seed == 12345
    assert f"final_size: {final_width}x{final_height}" in report
    assert f"downscaled_size: {downscaled.shape[2]}x{downscaled.shape[1]}" in report
    assert sampler.call_count == 1
    assert sampler.call_args.args[1:6] == (12345, 12, 2.5, "euler", "simple")
    assert sampler.call_args.kwargs["denoise"] == 0.16
    assert sampler.call_args.args[7] is negative
    reference = sampler.call_args.args[6][0][1]["reference_latents"][0]
    assert reference is vae.encoded
    assert positive[0][1] == {"keep": "positive"}
    assert negative[0][1] == {"keep": "negative"}
    for index in range(batch):
        torch.testing.assert_close(image[index], torch.full_like(image[index], 0.4 + index * 0.1))


@pytest.mark.parametrize("method", remaster.LatentUpscale.upscale_methods)
@pytest.mark.parametrize("width,height,final_width,final_height", [(111, 87, 104, 80), (9, 9, 64, 64)])
def test_factor8_interpolation_matches_previous_core_path(method, width, height, final_width, final_height):
    vae = _ImageVAE(8, 4)
    (image, _, latent, _, _), sampler, _, _ = _run_remaster(
        vae, width, height, denoise=0.0, reference=False, method=method, pattern=True
    )
    previous = remaster.LatentUpscale().upscale(
        {"samples": vae.encoded}, method, final_width, final_height, "disabled"
    )[0]
    torch.testing.assert_close(latent["samples"], previous["samples"], rtol=0, atol=0)
    assert image.shape == (1, final_height, final_width, 3)
    sampler.assert_not_called()


@pytest.mark.parametrize("blend,expected", [(0.0, 0.2), (0.25, 0.25), (1.0, 0.4)])
def test_moss_anchor_blending_keeps_aligned_target_size(blend, expected):
    (image, _, _, _, _), _, _, _ = _run_remaster(
        _ImageVAE(16, 128), 111, 87, blend=blend, reference=False
    )
    assert image.shape == (1, 80, 96, 3)
    torch.testing.assert_close(image, torch.full_like(image, expected), atol=1e-6, rtol=1e-6)


def test_moss_denoise_zero_skips_sampler_and_keeps_nonzero_encoded_latent():
    vae = _ImageVAE(16, 128)
    (image, _, latent, seed, report), sampler, _, _ = _run_remaster(
        vae, 96, 64, denoise=0.0, reference=False
    )
    sampler.assert_not_called()
    assert latent["samples"].shape == (1, 128, 4, 6)
    torch.testing.assert_close(latent["samples"], torch.full_like(latent["samples"], 0.2))
    torch.testing.assert_close(image, torch.full_like(image, 0.2))
    assert seed == 12345 and "sampled: False" in report


def test_encode_and_decode_compression_are_used_independently():
    vae = _ImageVAE(8, 4, decode_compression=16)
    (image, downscaled, latent, _, _), _, _, _ = _run_remaster(
        vae, 111, 87, denoise=0.0, reference=False
    )
    assert downscaled.shape == (1, 40, 56, 3)
    assert image.shape == (1, 80, 96, 3)
    assert latent["samples"].shape == (1, 4, 5, 6)


def test_latent_upscale_preserves_metadata_without_mutating_encoded_samples():
    vae = _ImageVAE(16, 128)
    samples = torch.full((1, 128, 2, 3), 0.7)
    mask = torch.ones(1, 32, 48)
    encoded = {"samples": samples, "noise_mask": mask, "batch_index": [7], "custom": object()}
    with patch.object(remaster.VAEEncodeTiled, "encode", return_value=(encoded,)):
        (_, _, latent, _, _), sampler, _, _ = _run_remaster(
            vae, 96, 64, denoise=0.0, reference=False
        )
    sampler.assert_not_called()
    assert latent is not encoded
    assert latent["samples"].shape == (1, 128, 4, 6)
    assert encoded["samples"] is samples and samples.shape == (1, 128, 2, 3)
    torch.testing.assert_close(samples, torch.full_like(samples, 0.7))
    for key in ("noise_mask", "batch_index", "custom"):
        assert latent[key] is encoded[key]



def test_moss_default_interpolation_handles_nonconstant_128_channel_latents():
    method = remaster.OnekoPixelAnchoredRemaster.INPUT_TYPES()["required"]["latent_upscale_method"][1]["default"]
    (image, _, latent, _, _), _, _, _ = _run_remaster(
        _ImageVAE(16, 128), 96, 64, method=method, pattern=True
    )
    assert image.shape == (1, 64, 96, 3)
    assert latent["samples"].shape == (1, 128, 4, 6)
    assert torch.isfinite(image).all() and torch.isfinite(latent["samples"]).all()
    assert image.max() > image.min()

if __name__ == "__main__":
    unittest.main()
