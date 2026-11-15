"""Release regressions for multi-stream sampling and safe Wan replacement."""
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from PIL import Image

from Oneko_ComfyUI_Nodes.nodes import noise_sampler_core as core
from Oneko_ComfyUI_Nodes.nodes import wan22_video_toolkit as wan


@pytest.mark.parametrize("callback_layout", ["nested", "packed", "dense"])
def test_x0_preserves_streams_and_processing_order(callback_layout):
    nested = core.comfy.nested_tensor.NestedTensor
    streams = [torch.arange(16.).reshape(1, 4, 2, 2), torch.arange(6.).reshape(1, 2, 3)]
    samples = streams[0] if callback_layout == "dense" else nested(streams)
    x0 = core.comfy.utils.pack_latents(streams)[0] if callback_layout == "packed" else samples

    def process_out(value):
        if callback_layout == "dense":
            assert not value.is_nested
            return value * 2
        assert value.is_nested
        return nested([value.unbind()[0] * 2, value.unbind()[1] * 3])

    def prepare_callback(model, steps, output):
        def callback(step, predicted, current, total):
            output["x0"] = predicted
        return callback

    def sample(noise, latent, sampler, sigmas, callback=None, **kwargs):
        assert kwargs["seed"] == 123
        callback(0, x0, latent, 1)
        return samples

    patcher = SimpleNamespace(model=SimpleNamespace(process_latent_out=process_out))
    guider = SimpleNamespace(model_patcher=patcher, sample=sample)
    with (
        mock.patch.object(core.comfy.sample, "fix_empty_latent_channels", side_effect=lambda model, value, *args: value),
        mock.patch.object(core.latent_preview, "prepare_callback", side_effect=prepare_callback),
        mock.patch.object(core.comfy.model_management, "intermediate_device", return_value="cpu"),
    ):
        out, denoised, seed = core.sample_custom_advanced(
            guider, object(), torch.tensor([1., 0.]), {"samples": samples},
            core.OnekoEmptyNoise(), 123,
        )
    assert seed == 123
    if callback_layout == "dense":
        torch.testing.assert_close(out["samples"], streams[0], rtol=0, atol=0)
        torch.testing.assert_close(denoised["samples"], streams[0] * 2, rtol=0, atol=0)
    else:
        assert out["samples"].is_nested and denoised["samples"].is_nested
        for index, original in enumerate(streams):
            torch.testing.assert_close(out["samples"].unbind()[index], original, rtol=0, atol=0)
            torch.testing.assert_close(
                denoised["samples"].unbind()[index], original * (index + 2), rtol=0, atol=0,
            )


@pytest.fixture
def saved_wan_run(tmp_path):
    with mock.patch.object(wan.folder_paths, "get_output_directory", return_value=str(tmp_path)):
        store = wan.OnekoWan22SegmentStore()
        images = torch.zeros(3, 16, 16, 3)
        store.store(images, "release_test", 0, 16, False, True, 1, 2, 3, "", "", "", "", "{}")
        store.store(images, "release_test", 1, 16, True, True, 4, 5, 6, "", "", "", "", "{}")
        yield wan._run_dir("release_test")


def snapshot(run_dir):
    return {str(path.relative_to(run_dir)): path.read_bytes()
            for path in run_dir.rglob("*") if path.is_file()}


@pytest.mark.parametrize("segment_index", [0, 1])
@pytest.mark.parametrize("invalid_input", ["manifest", "empty_after_drop", "dimensions"])
def test_invalid_segment_replacement_preserves_saved_run(saved_wan_run, segment_index, invalid_input):
    before = snapshot(saved_wan_run)
    images = torch.ones(3, 16, 16, 3)
    manifest, drop_first = "{}", False
    if invalid_input == "manifest":
        manifest = "not json"
    elif invalid_input == "empty_after_drop":
        images, drop_first = images[:1], True
    else:
        images = images[:, :0]
    with pytest.raises(RuntimeError):
        wan.OnekoWan22SegmentStore().store(
            images, "release_test", segment_index, 16, drop_first, True,
            7, 8, 9, "", "", "", "", manifest,
        )
    assert snapshot(saved_wan_run) == before


@pytest.mark.parametrize("failure", ["dimensions", "encoder", "no_output"])
def test_failed_video_replacement_preserves_previous_export(saved_wan_run, failure):
    videos = saved_wan_run / "videos"
    videos.mkdir()
    target = videos / "final.mp4"
    original = b"previous successful video"
    target.write_bytes(original)
    if failure == "dimensions":
        Image.new("RGB", (32, 16)).save(saved_wan_run / "frames/frame_000001.png")
        with pytest.raises(RuntimeError, match="size mismatch"):
            wan.OnekoWan22FrameSequenceAssembler().assemble("release_test", 16, "final", "libx264", 8, True)
    else:
        import imageio.v2 as imageio
        with mock.patch.object(imageio, "get_writer") as get_writer:
            if failure == "encoder":
                get_writer.return_value.append_data.side_effect = RuntimeError("encoder failed")
            with pytest.raises(RuntimeError, match="encoder"):
                wan.OnekoWan22FrameSequenceAssembler().assemble("release_test", 16, "final", "libx264", 8, True)
            get_writer.return_value.close.assert_called_once()
    assert target.read_bytes() == original
    assert sorted(path.name for path in videos.iterdir()) == ["final.mp4"]


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_invalid_video_fps_does_not_modify_run(saved_wan_run, fps):
    before = snapshot(saved_wan_run)
    with pytest.raises(ValueError, match="FPS"):
        wan.OnekoWan22FrameSequenceAssembler().assemble("release_test", fps, "final", "libx264", 8, True)
    assert snapshot(saved_wan_run) == before
