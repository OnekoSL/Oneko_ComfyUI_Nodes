import sys
import unittest
from pathlib import Path
from unittest import mock


from tests.paths import COMFY_ROOT, ONEKO_ROOT
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

from Oneko_ComfyUI_Nodes.nodes import ollama_prompt_refiner as refiner


class Wan22PromptProfileTests(unittest.TestCase):
    def values(self):
        return {
            "base_prompt": "Locked medium shot, warm cinematic light, natural color, stable camera.",
            "foreground_prompt": (
                "A sword dancer steps left, turns smoothly, swings once, and settles into a balanced guard pose. "
                "Her expression and clothing remain consistent throughout the continuous action."
            ),
            "background_prompt": (
                "A dusty arena remains spatially stable while fine dust follows her feet and settles naturally."
            ),
            "negative": "bad anatomy",
            "report": "Converted the source into one continuous video shot.",
        }

    def test_profile_is_selectable(self):
        self.assertIn("wan2_2_video", refiner.TARGET_PROFILES)
        self.assertEqual(refiner._normalize_target_profile("wan2_2_video"), "wan2_2_video")

    def test_prompt_order_is_subject_scene_camera(self):
        positive, _negative, _report, base, foreground, background = refiner._postprocess_result(
            self.values(), "wan2_2_video", "dancer dusty arena", "", 430
        )
        self.assertLess(positive.index(foreground), positive.index(background))
        self.assertLess(positive.index(background), positive.index(base))
        self.assertIn("settles into a balanced guard pose", positive)

    def test_video_negative_baseline_is_added(self):
        _positive, negative, *_rest = refiner._postprocess_result(
            self.values(), "wan2_2_video", "dancer dusty arena", "", 430
        )
        for term in ("flicker", "temporal jitter", "identity drift", "frozen motion", "camera shake"):
            self.assertIn(term, negative)

    def test_instructions_require_action_camera_and_temporal_consistency(self):
        instructions = refiner._target_profile_instructions("wan2_2_video", 430).lower()
        self.assertIn("continuous temporal action", instructions)
        self.assertIn("camera", instructions)
        self.assertIn("temporally consistent", instructions)

    def test_unreachable_ollama_is_a_hard_error(self):
        error = refiner.urllib.error.URLError("offline")
        with mock.patch.object(refiner.urllib.request, "urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "could not reach Ollama"):
                refiner._request_ollama(
                    "http://127.0.0.1:11434",
                    "test-model",
                    "prompt",
                    0,
                    0.0,
                    1.0,
                    1,
                    2048,
                )


if __name__ == "__main__":
    unittest.main()
