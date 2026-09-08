import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


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


if __name__ == "__main__":
    unittest.main()
