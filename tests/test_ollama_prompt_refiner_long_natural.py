import sys
import unittest
from pathlib import Path


from tests.paths import COMFY_ROOT, ONEKO_ROOT
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

from Oneko_ComfyUI_Nodes.nodes import ollama_prompt_refiner as refiner


def section_to_target(kind, target_words):
    templates = {
        "base": (
            "The visual treatment detail {index} uses cinematic lighting, balanced contrast, natural color, "
            "precise focus, and believable material texture."
        ),
        "foreground": (
            "The main subject detail {index} shows clear anatomy, deliberate posture, layered fabric, reflective "
            "metal, expressive movement, and readable physical contact."
        ),
        "background": (
            "The environment detail {index} places concrete architecture, practical objects, atmospheric light, "
            "and distant forms across clearly visible spatial depth."
        ),
    }
    sentences = []
    index = 1
    while True:
        sentence = templates[kind].format(index=index)
        candidate = " ".join((*sentences, sentence))
        if sentences and refiner._word_count(candidate) > target_words:
            break
        sentences.append(sentence)
        index += 1
    return " ".join(sentences)


class LongNaturalPromptTests(unittest.TestCase):
    def setUp(self):
        self.base = section_to_target("base", 65)
        self.foreground = section_to_target("foreground", 155)
        self.background = section_to_target("background", 115)
        self.values = {
            "base_prompt": self.base,
            "foreground_prompt": self.foreground,
            "background_prompt": self.background,
            "negative": "blurry, distorted geometry",
            "report": "Built a detailed natural image description.",
        }

    def test_shared_targets_are_subject_focused(self):
        self.assertEqual(refiner.Z_IMAGE_POSITIVE_WORD_RANGE, (300, 360))
        self.assertEqual(refiner.KREA2_POSITIVE_WORD_RANGE, (300, 360))
        self.assertEqual(refiner.Z_IMAGE_BASE_WORD_RANGE, (55, 70))
        self.assertEqual(refiner.Z_IMAGE_FOREGROUND_WORD_RANGE, (140, 165))
        self.assertEqual(refiner.Z_IMAGE_BACKGROUND_WORD_RANGE, (105, 125))
        self.assertEqual(refiner.KREA2_BASE_WORD_RANGE, (55, 70))
        self.assertEqual(refiner.KREA2_FOREGROUND_WORD_RANGE, (140, 165))
        self.assertEqual(refiner.KREA2_BACKGROUND_WORD_RANGE, (105, 125))

    def test_krea2_keeps_a_long_valid_response(self):
        positive, _negative, _report, base, foreground, background = refiner._postprocess_result(
            self.values,
            "krea2",
            "subject architecture material lighting",
            "",
            430,
        )
        self.assertGreater(refiner._word_count(foreground), 85)
        self.assertGreater(refiner._word_count(background), 60)
        self.assertGreaterEqual(refiner._word_count(positive), 300)
        self.assertLessEqual(refiner._word_count(positive), 360)
        self.assertEqual(positive, refiner._join_positive_parts("krea2", base, foreground, background))
        self.assertTrue(positive.startswith(foreground + "\n\n"))
        self.assertEqual(len(positive.split("\n\n")), 2)

    def test_z_image_keeps_long_prose_and_its_section_order(self):
        positive, negative, _report, base, foreground, background = refiner._postprocess_result(
            self.values,
            "z_image",
            "subject architecture material lighting",
            "",
            430,
        )
        self.assertEqual(negative, "")
        self.assertGreaterEqual(refiner._word_count(positive), 300)
        self.assertLessEqual(refiner._word_count(positive), 360)
        self.assertTrue(positive.startswith(foreground))
        self.assertLess(positive.index(foreground), positive.index(background))
        self.assertLess(positive.index(background), positive.index(base))
        self.assertEqual(len(positive.split("\n\n")), 2)

    def test_z_image_preserves_style_anchor_in_closing_base(self):
        positive, negative, _report, base, foreground, _background = refiner._postprocess_result(
            self.values,
            "z_image",
            "subject architecture material lighting",
            "fixed LoRA_trigger",
            430,
        )
        self.assertEqual(negative, "")
        self.assertTrue(positive.startswith(foreground + "\n\n"))
        self.assertTrue(base.startswith("fixed LoRA_trigger,"))

    def test_spatial_sentences_survive_after_a_long_krea2_response(self):
        result = refiner._postprocess_result(
            self.values,
            "krea2",
            "subject architecture material lighting",
            "fixed style anchor",
            430,
        )
        result = refiner._apply_spatial_result(
            result,
            "krea2",
            left="a red fox beside a stone",
            top="golden clouds above the tower",
        )
        self.assertTrue(result[0].startswith(result[4] + "\n\n"))
        self.assertTrue(result[3].startswith("fixed style anchor,"))
        self.assertIn("On the left, a red fox beside a stone.", result[0])
        self.assertIn("Across the top, golden clouds above the tower.", result[5])


if __name__ == "__main__":
    unittest.main()
