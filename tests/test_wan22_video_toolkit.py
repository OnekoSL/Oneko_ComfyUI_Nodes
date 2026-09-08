import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


from tests.paths import COMFY_ROOT, ONEKO_ROOT
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

from Oneko_ComfyUI_Nodes.nodes import wan22_video_toolkit as toolkit


class FakeWanVAE:
    def __init__(self):
        self.last_image = None

    def encode(self, image):
        self.last_image = image
        batch, height, width, _channels = image.shape
        return torch.ones((batch, 48, 1, height // 16, width // 16), dtype=image.dtype)


class Wan22VideoSettingsTests(unittest.TestCase):
    def test_balanced_five_seconds_at_16_fps_produces_81_frames(self):
        settings = toolkit._build_settings(
            "text_to_video", "balanced", "landscape", 5.0, 16.0, 960, 544
        )
        self.assertEqual((settings["width"], settings["height"]), (960, 544))
        self.assertEqual(settings["length"], 81)
        self.assertAlmostEqual(settings["actual_duration"], 81 / 16)

    def test_orientation_presets_are_valid_wan_dimensions(self):
        portrait = toolkit._build_settings(
            "image_to_video", "quality", "portrait", 5.0, 16.0, 960, 544
        )
        square = toolkit._build_settings(
            "text_to_video", "draft", "square", 5.0, 16.0, 960, 544
        )
        self.assertEqual((portrait["width"], portrait["height"]), (704, 1280))
        self.assertEqual((square["width"], square["height"]), (512, 512))
        for settings in (portrait, square):
            self.assertEqual(settings["width"] % 32, 0)
            self.assertEqual(settings["height"] % 32, 0)
            self.assertEqual((settings["length"] - 1) % 4, 0)

    def test_custom_dimensions_are_snapped_and_oriented(self):
        settings = toolkit._build_settings(
            "text_to_video", "custom", "landscape", 1.0, 16.0, 511, 1001
        )
        self.assertEqual((settings["width"], settings["height"]), (992, 512))
        self.assertEqual(settings["length"], 17)

    def test_invalid_duration_and_fps_fail_early(self):
        with self.assertRaisesRegex(ValueError, "duration"):
            toolkit._build_settings("text_to_video", "draft", "landscape", 0, 16, 640, 352)
        with self.assertRaisesRegex(ValueError, "FPS"):
            toolkit._build_settings("text_to_video", "draft", "landscape", 1, 0, 640, 352)


class Wan22LatentTests(unittest.TestCase):
    def setUp(self):
        self.node = toolkit.OnekoWan22TI2VLatent()

    def settings(self, mode, width=640, height=352, length=17):
        return {
            "mode": mode,
            "quality": "draft",
            "orientation": "landscape",
            "width": width,
            "height": height,
            "length": length,
            "fps": 16.0,
            "requested_duration": 1.0,
            "actual_duration": length / 16.0,
        }

    def test_t2v_ignores_connected_image_and_has_no_noise_mask(self):
        image = torch.rand((1, 80, 40, 3))
        latent, report = self.node.create(
            FakeWanVAE(), self.settings("text_to_video"), "center_crop", image
        )
        self.assertEqual(tuple(latent["samples"].shape), (1, 48, 5, 22, 40))
        self.assertNotIn("noise_mask", latent)
        self.assertIn("ignored", report)

    def test_i2v_requires_start_image(self):
        with self.assertRaisesRegex(RuntimeError, "requires a connected start_image"):
            self.node.create(FakeWanVAE(), self.settings("image_to_video"), "center_crop")

    def test_i2v_center_crop_prepares_target_without_stretching_interface(self):
        vae = FakeWanVAE()
        image = torch.rand((1, 300, 200, 3))
        latent, report = self.node.create(
            vae, self.settings("image_to_video"), "center_crop", image
        )
        self.assertEqual(tuple(vae.last_image.shape), (1, 352, 640, 3))
        self.assertEqual(tuple(latent["samples"].shape), (1, 48, 5, 22, 40))
        self.assertEqual(tuple(latent["noise_mask"].shape), (1, 1, 5, 22, 40))
        self.assertEqual(float(latent["noise_mask"][:, :, 0].max()), 0.0)
        self.assertIn("center_crop", report)

    def test_i2v_pad_preserves_image_and_adds_borders(self):
        prepared = toolkit._prepare_image(torch.ones((1, 100, 100, 3)), 640, 352, "pad")
        self.assertEqual(tuple(prepared.shape), (1, 352, 640, 3))
        self.assertEqual(float(prepared[:, :, 0].max()), 0.0)
        self.assertGreater(float(prepared[:, :, 320].mean()), 0.9)


class Wan22ManifestTests(unittest.TestCase):
    def test_manifest_is_reproducible_and_filename_safe(self):
        settings = toolkit._build_settings(
            "text_to_video", "balanced", "landscape", 5.0, 16.0, 960, 544
        )
        manifest_text, prefix = toolkit.OnekoWan22RunManifest().compose(
            settings,
            "wan/wan2.2_ti2v_5B_fp16.safetensors",
            12,
            34,
            25,
            5.0,
            "dpmpp_2m",
            "bong_tangent",
            8.0,
            "positive",
            "negative",
        )
        manifest = json.loads(manifest_text)
        self.assertEqual(manifest["schema"], "oneko.wan22.run.v1")
        self.assertEqual(manifest["frames"], 81)
        self.assertEqual(manifest["sampling_seed"], 34)
        self.assertEqual(
            prefix,
            "wan22/wan2.2_ti2v_5B_fp16_text_to_video_960x544_f81_ps12_ss34",
        )


class Wan22ContinuationTests(unittest.TestCase):
    def balanced_portrait_settings(self):
        return toolkit._build_settings(
            "image_to_video", "balanced", "portrait", 5.0, 16.0, 544, 960
        )

    def base_manifest(self):
        settings = self.balanced_portrait_settings()
        return toolkit.OnekoWan22RunManifest().compose(
            settings,
            "wan/wan2.2_ti2v_5B_fp16.safetensors",
            44004,
            55005,
            25,
            5.0,
            "res_2m",
            "bong_tangent",
            8.0,
            "base positive",
            "base negative",
        )[0]

    def test_one_extension_has_161_frames_and_ten_seconds_motion(self):
        plan = toolkit._build_continuation_plan(self.balanced_portrait_settings(), 1)
        self.assertEqual(plan["segment_frames"], 81)
        self.assertEqual(plan["total_frames"], 161)
        self.assertEqual(plan["motion_duration"], 10.0)
        self.assertEqual(plan["container_duration"], 161 / 16)

    def test_ten_extensions_have_881_frames_and_conservative_ram_estimate(self):
        plan = toolkit._build_continuation_plan(self.balanced_portrait_settings(), 10)
        self.assertEqual(plan["total_frames"], 881)
        self.assertEqual(plan["motion_duration"], 55.0)
        self.assertGreater(plan["estimated_frame_ram_gb"], 5.0)
        self.assertLess(plan["estimated_frame_ram_gb"], 5.2)
        self.assertIn("identity drift", toolkit._continuation_report(plan))

    def test_extension_count_is_limited_to_one_through_ten(self):
        for count in (0, 11):
            with self.assertRaisesRegex(RuntimeError, "between 1 and 10"):
                toolkit._build_continuation_plan(self.balanced_portrait_settings(), count)

    def test_seam_removal_keeps_only_one_copy_of_boundary_frame(self):
        first = torch.arange(81).reshape(81, 1, 1, 1).float()
        extension = torch.arange(80, 161).reshape(81, 1, 1, 1).float()
        joined = torch.cat((first, extension[1:]), dim=0)
        self.assertEqual(joined.shape[0], 161)
        self.assertEqual(joined[:, 0, 0, 0].tolist(), list(range(161)))

    def test_record_and_final_manifest_preserve_iteration_seeds_and_prompts(self):
        plan = toolkit._build_continuation_plan(self.balanced_portrait_settings(), 1)
        log_text = toolkit.OnekoWan22ContinuationRecord().append(
            plan,
            "[]",
            0,
            33004,
            44005,
            55006,
            "current end frame caption",
            "continued positive",
            "continued negative",
        )[0]
        records = json.loads(log_text)
        self.assertEqual(records[0]["segment_index"], 1)
        self.assertEqual(records[0]["vision_seed"], 33004)
        self.assertEqual(records[0]["prompt_seed"], 44005)
        self.assertEqual(records[0]["sampling_seed"], 55006)

        manifest_text, prefix = toolkit.OnekoWan22ContinuationManifest().compose(
            self.base_manifest(), plan, log_text
        )
        manifest = json.loads(manifest_text)
        self.assertEqual(manifest["schema"], "oneko.wan22.continuation.v1")
        self.assertEqual(manifest["continuation"]["total_frames"], 161)
        self.assertEqual(len(manifest["extensions"]), 1)
        self.assertIn("_ext1_f161_", prefix)

    def test_manifest_rejects_incomplete_loop_log(self):
        plan = toolkit._build_continuation_plan(self.balanced_portrait_settings(), 2)
        with self.assertRaisesRegex(RuntimeError, "expected 2 record"):
            toolkit.OnekoWan22ContinuationManifest().compose(
                self.base_manifest(), plan, "[]"
            )


class Wan22SegmentFileTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.output_patch = mock.patch.object(
            toolkit.folder_paths, "get_output_directory", return_value=self.tempdir.name
        )
        self.output_patch.start()

    def tearDown(self):
        self.output_patch.stop()
        self.tempdir.cleanup()

    def images(self, count, value_offset=0):
        values = torch.linspace(0.05, 0.95, steps=count).reshape(count, 1, 1, 1)
        return values.repeat(1, 16, 16, 3) + (value_offset / 255.0)

    def test_segment_zero_and_one_store_33_frames_without_duplicate_seam(self):
        store = toolkit.OnekoWan22SegmentStore()
        run_id, run_dir, next_index, stored, total, _last, segment_json, _report = store.store(
            self.images(17),
            "smoke_run",
            0,
            16,
            False,
            True,
            33003,
            44004,
            55005,
            "original caption",
            "first caption",
            "positive 0",
            "negative 0",
            "{}",
        )
        self.assertEqual(run_id, "smoke_run")
        self.assertEqual(next_index, 1)
        self.assertEqual(stored, 17)
        self.assertEqual(total, 17)
        self.assertEqual(json.loads(segment_json)["last_frame_index"], 16)

        run_id, _run_dir, next_index, stored, total, _last, segment_json, _report = store.store(
            self.images(17, value_offset=1),
            run_id,
            1,
            16,
            True,
            True,
            33004,
            44005,
            55006,
            "original caption",
            "second caption",
            "positive 1",
            "negative 1",
            "{}",
        )
        self.assertEqual(next_index, 2)
        self.assertEqual(stored, 16)
        self.assertEqual(total, 33)
        segment = json.loads(segment_json)
        self.assertEqual(segment["first_frame_index"], 17)
        self.assertEqual(segment["last_frame_index"], 32)

        frames = sorted((Path(run_dir) / "frames").glob("frame_*.png"))
        self.assertEqual(len(frames), 33)
        self.assertEqual(frames[16].name, "frame_000016.png")
        self.assertEqual(frames[17].name, "frame_000017.png")

    def test_loader_returns_next_segment_and_last_frame_state(self):
        store = toolkit.OnekoWan22SegmentStore()
        store.store(
            self.images(17),
            "loadable_run",
            0,
            16,
            False,
            True,
            1,
            2,
            3,
            "identity anchor",
            "base caption",
            "positive",
            "negative",
            "{}",
        )
        last_frame, run_id, next_index, total, original, current, manifest_json, report = (
            toolkit.OnekoWan22SegmentLoader().load("loadable_run", -1)
        )
        self.assertEqual(tuple(last_frame.shape), (1, 16, 16, 3))
        self.assertEqual(run_id, "loadable_run")
        self.assertEqual(next_index, 1)
        self.assertEqual(total, 17)
        self.assertEqual(original, "identity anchor")
        self.assertEqual(current, "base caption")
        self.assertEqual(json.loads(manifest_json)["total_frames"], 17)
        self.assertIn("next segment 1", report)

    def test_rerunning_a_segment_replaces_only_the_tail(self):
        store = toolkit.OnekoWan22SegmentStore()
        store.store(self.images(17), "rerun", 0, 16, False, True, 1, 2, 3, "", "", "", "", "{}")
        store.store(self.images(17), "rerun", 1, 16, True, True, 4, 5, 6, "", "", "", "", "{}")
        store.store(self.images(17), "rerun", 1, 16, True, True, 7, 8, 9, "", "", "", "", "{}")
        run_dir = toolkit._run_dir("rerun")
        frames = sorted((run_dir / "frames").glob("frame_*.png"))
        self.assertEqual(len(frames), 33)
        run_manifest = json.loads((run_dir / "manifests" / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(run_manifest["total_frames"], 33)
        segment = json.loads((run_dir / "manifests" / "segment_001.json").read_text(encoding="utf-8"))
        self.assertEqual(segment["prompt_seed"], 8)

    def test_assembler_creates_mp4_and_rejects_gaps(self):
        store = toolkit.OnekoWan22SegmentStore()
        store.store(self.images(17), "assemble_run", 0, 16, False, True, 1, 2, 3, "", "", "", "", "{}")
        store.store(self.images(17), "assemble_run", 1, 16, True, True, 4, 5, 6, "", "", "", "", "{}")

        video_path, total, duration, manifest_json, report = toolkit.OnekoWan22FrameSequenceAssembler().assemble(
            "assemble_run", 16, "final", "libx264", 8, True
        )
        self.assertTrue(Path(video_path).exists())
        self.assertEqual(total, 33)
        self.assertAlmostEqual(duration, 33 / 16)
        self.assertEqual(json.loads(manifest_json)["total_frames"], 33)
        self.assertIn("assembled", report)

        (toolkit._run_dir("assemble_run") / "frames" / "frame_000005.png").unlink()
        with self.assertRaisesRegex(RuntimeError, "not contiguous"):
            toolkit.OnekoWan22FrameSequenceAssembler().assemble(
                "assemble_run", 16, "broken", "libx264", 8, True
            )

    def test_filesystem_state_nodes_force_cache_refresh(self):
        for node_class in (
            toolkit.OnekoWan22SegmentLoader,
            toolkit.OnekoWan22SegmentStore,
            toolkit.OnekoWan22FrameSequenceAssembler,
        ):
            value = node_class.IS_CHANGED()
            self.assertNotEqual(value, value)


if __name__ == "__main__":
    unittest.main()
