"""Local preview selection, native decoder parity and overlapping samplers."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import comfy.latent_formats
from comfy.nested_tensor import NestedTensor
from comfy.cli_args import LatentPreviewMethod
from Oneko_ComfyUI_Nodes.nodes import preview_core as preview
from Oneko_ComfyUI_Nodes.nodes import noise_sampler_core as noise_core


def model():
    return SimpleNamespace(load_device=torch.device("cpu"),
                           model=SimpleNamespace(latent_format=comfy.latent_formats.SD15(),
                                                 process_latent_out=lambda value: value))


@pytest.mark.parametrize("method", [None, "", "default"])
def test_default_delegates_without_mutating_active_core_setting(monkeypatch, method):
    sentinel = object()
    native = Mock(return_value=sentinel)
    monkeypatch.setattr(preview.latent_preview, "prepare_callback", native)
    monkeypatch.setattr(preview.latent_preview.args, "preview_method", LatentPreviewMethod.TAESD)
    patcher, output = model(), {}
    assert preview.prepare_preview_callback(patcher, 7, output, method) is sentinel
    native.assert_called_once_with(patcher, 7, output)
    assert preview.latent_preview.args.preview_method is LatentPreviewMethod.TAESD


@pytest.mark.parametrize("method", ["none", "auto", "latent2rgb", "taesd"])
def test_local_selection_matches_native_decoder_and_rgb_pixels(monkeypatch, method):
    method_enum = LatentPreviewMethod.from_string(method)
    latent_format = comfy.latent_formats.SD15()
    monkeypatch.setattr(preview.folder_paths, "get_filename_list", lambda *args: [])
    monkeypatch.setattr(preview.folder_paths, "get_full_path", lambda *args: None)
    monkeypatch.setattr(preview.latent_preview.args, "preview_method", method_enum)
    native = preview.latent_preview.get_previewer("cpu", latent_format)
    local = preview._get_previewer("cpu", latent_format, method_enum)
    assert type(local) is type(native)
    if local is not None:
        samples = torch.linspace(-1, 1, 4 * 2 * 3).reshape(1, 4, 2, 3)
        assert local.decode_latent_to_preview(samples).tobytes() == native.decode_latent_to_preview(samples).tobytes()


@pytest.mark.parametrize("video", [False, True])
def test_taesd_uses_native_image_or_video_decoder_without_global_change(monkeypatch, video):
    fmt = model().model.latent_format
    fmt.taesd_decoder_name = preview.latent_preview.VIDEO_TAES[0] if video else "taesd_decoder"
    fmt.compile_preview = True
    filename = fmt.taesd_decoder_name + ".safetensors"
    monkeypatch.setattr(preview.folder_paths, "get_filename_list", lambda *args: ["unrelated", filename])
    path_lookup = Mock(return_value="local-decoder.safetensors")
    monkeypatch.setattr(preview.folder_paths, "get_full_path", path_lookup)
    monkeypatch.setattr(preview.latent_preview.args, "preview_method", LatentPreviewMethod.NoPreviews)
    decoder = SimpleNamespace(first_stage_model=SimpleNamespace(show_progress_bar=True))
    decoder.to = Mock(return_value=decoder)
    taesd = Mock(return_value=decoder)
    vae = Mock(return_value=decoder)
    load = Mock(return_value={"local weights": True})
    image_previewer, video_previewer = Mock(), Mock()
    monkeypatch.setattr(preview.latent_preview, "TAESD", taesd)
    monkeypatch.setattr(preview.latent_preview, "VAE", vae)
    monkeypatch.setattr(preview.comfy.utils, "load_torch_file", load)
    monkeypatch.setattr(preview.latent_preview, "TAESDPreviewerImpl", image_previewer)
    monkeypatch.setattr(preview.latent_preview, "TAEHVPreviewerImpl", video_previewer)
    actual = preview._get_previewer("cpu", fmt, LatentPreviewMethod.TAESD)
    path_lookup.assert_called_once_with("vae_approx", filename)
    if video:
        load.assert_called_once_with("local-decoder.safetensors")
        vae.assert_called_once_with(load.return_value)
        video_previewer.assert_called_once_with(decoder, compile_preview=True)
        assert decoder.first_stage_model.show_progress_bar is False
        assert actual is video_previewer.return_value
        taesd.assert_not_called()
    else:
        taesd.assert_called_once_with(None, "local-decoder.safetensors", latent_channels=4)
        decoder.to.assert_called_once_with("cpu")
        image_previewer.assert_called_once_with(decoder)
        assert actual is image_previewer.return_value
        vae.assert_not_called()
    assert preview.latent_preview.args.preview_method is LatentPreviewMethod.NoPreviews


def test_missing_rgb_factors_and_decoder_produce_no_preview(monkeypatch):
    fmt = model().model.latent_format
    fmt.latent_rgb_factors = None
    fmt.taesd_decoder_name = None
    assert preview._get_previewer("cpu", fmt, LatentPreviewMethod.TAESD) is None


def test_callback_keeps_full_nested_x0_but_previews_only_first_stream(monkeypatch):
    progress = Mock()
    monkeypatch.setattr(preview.comfy.utils, "ProgressBar", Mock(return_value=progress))
    decoder = Mock()
    monkeypatch.setattr(preview, "_get_previewer", lambda *args: decoder)
    output = {}
    callback = preview.prepare_preview_callback(model(), 3, output, "latent2rgb")
    samples = NestedTensor([torch.ones(1, 4, 2, 2), torch.zeros(1, 2, 3)])
    callback(1, samples, samples, 3)
    assert output["x0"] is samples
    decoder.decode_latent_to_preview_image.assert_called_once_with("JPEG", samples.tensors[0])
    progress.update_absolute.assert_called_once_with(2, 3, decoder.decode_latent_to_preview_image.return_value)


@pytest.mark.parametrize("fail_first", [False, True])
def test_overlapping_samplers_keep_previews_x0_and_global_settings_isolated(monkeypatch, fail_first):
    active_method = LatentPreviewMethod.TAESD
    monkeypatch.setattr(preview.latent_preview.args, "preview_method", active_method)
    setter = Mock(side_effect=AssertionError("global preview mutation"))
    monkeypatch.setattr(preview.latent_preview, "set_preview_method", setter)
    monkeypatch.setattr(noise_core.comfy.sample, "fix_empty_latent_channels", lambda m, value, *args: value)
    monkeypatch.setattr(noise_core.comfy.model_management, "intermediate_device", lambda: "cpu")
    bars = []

    def progress(steps):
        result = Mock()
        bars.append(result)
        return result

    monkeypatch.setattr(preview.comfy.utils, "ProgressBar", progress)
    decoder = Mock()
    monkeypatch.setattr(preview, "_get_previewer",
                        lambda device, fmt, method: None if method is LatentPreviewMethod.NoPreviews else decoder)
    barrier = Barrier(2)

    def run(index):
        latent = torch.full((1, 4, 2, 2), float(index + 1))

        def sample(noise, values, sampler, sigmas, callback=None, **kwargs):
            barrier.wait(timeout=10)
            assert preview.latent_preview.args.preview_method is active_method
            callback(0, values + 2, values, 1)
            if fail_first and index == 0:
                raise RuntimeError("isolated sampling failure")
            return values + 1

        guider = SimpleNamespace(model_patcher=model(), sample=sample)
        return noise_core.sample_custom_advanced(
            guider, object(), torch.tensor([1., 0.]), {"samples": latent},
            noise_core.OnekoEmptyNoise(), 40 + index,
            preview_method="latent2rgb" if index == 0 else "none",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, index) for index in range(2)]
        for index, future in enumerate(futures):
            if index == 0 and fail_first:
                with pytest.raises(RuntimeError, match="isolated sampling failure"):
                    future.result(timeout=15)
            else:
                out, denoised, seed = future.result(timeout=15)
                torch.testing.assert_close(out["samples"], torch.full((1, 4, 2, 2), index + 2.))
                torch.testing.assert_close(denoised["samples"], torch.full((1, 4, 2, 2), index + 3.))
                assert seed == 40 + index
    assert preview.latent_preview.args.preview_method is active_method
    setter.assert_not_called()
    assert decoder.decode_latent_to_preview_image.call_count == 1
    assert len(bars) == 2 and all(bar.update_absolute.call_count == 1 for bar in bars)
    previews = [bar.update_absolute.call_args.args[2] for bar in bars]
    assert previews.count(None) == 1


def test_previewer_creation_failure_does_not_touch_global_state(monkeypatch):
    monkeypatch.setattr(preview.latent_preview.args, "preview_method", LatentPreviewMethod.Auto)
    monkeypatch.setattr(preview, "_get_previewer", Mock(side_effect=RuntimeError("decoder failed")))
    with pytest.raises(RuntimeError, match="decoder failed"):
        preview.prepare_preview_callback(model(), 3, preview_method="taesd")
    assert preview.latent_preview.args.preview_method is LatentPreviewMethod.Auto
