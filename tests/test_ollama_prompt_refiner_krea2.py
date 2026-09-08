import sys
import unittest
from pathlib import Path


from tests.paths import COMFY_ROOT, ONEKO_ROOT
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

from Oneko_ComfyUI_Nodes.nodes import ollama_prompt_refiner as refiner


class Krea2PromptProfileTests(unittest.TestCase):
    def values(self):
        return {
            "base_prompt": (
                "score_9, masterpiece, A cinematic medium shot uses warm side light, muted copper and blue "
                "colors, shallow focal depth, and realistic glass reflections."
            ),
            "foreground_prompt": (
                "foreground_prompt: One copper-haired archivist opens a worn leather book beside a compact "
                "brass machine. She holds a palm-sized blue crystal above the page while its glow reflects "
                "across her fingers, silver buttons, damp coat seams, and three clear glass bottles. "
                "A small label on the machine reads \"ARCHIVE_7\" in white letters."
            ),
            "background_prompt": (
                "background_prompt: A narrow wooden workshop surrounds her. Tall rain-streaked windows stand "
                "behind the table, with blurred city rooftops beyond them. A copper lamp to her left creates "
                "warm contact shadows beneath the tools and books."
            ),
            "negative": "bad anatomy",
            "report": "Kept the archivist, crystal, workshop, and copper lighting.",
        }

    def result(self, style_anchor="mgrtt style"):
        return refiner._postprocess_result(
            self.values(),
            "krea2",
            "archivist crystal workshop rain glass bottles copper lamp",
            style_anchor,
            430,
        )

    def test_profile_is_selectable(self):
        self.assertIn("krea2", refiner.TARGET_PROFILES)
        self.assertEqual(refiner._normalize_target_profile("krea2"), "krea2")

    def test_instructions_match_krea2_natural_language_conditioning(self):
        instructions = refiner._target_profile_instructions("krea2", 430).lower()
        for term in ("natural english", "quantity", "shape", "material", "texture", "spatial relationships"):
            self.assertIn(term, instructions)
        for target in ("300 to 360", "55 to 70", "140 to 165", "105 to 125"):
            self.assertIn(target, instructions)
        self.assertIn("quote its exact content", instructions)
        self.assertIn("must begin exactly", instructions)
        candidate_context = refiner._build_candidate_context(
            "krea2", "archivist crystal workshop rain", "watercolor style"
        )
        example = refiner._candidate_few_shot_example(
            "krea2", "archivist crystal workshop rain", "watercolor style"
        )
        self.assertIn("For Krea 2", candidate_context)
        self.assertEqual(example, "")

    def test_natural_profile_prompts_do_not_embed_unrelated_example_motifs(self):
        for profile in ("anima", "krea2"):
            with self.subTest(profile=profile):
                prompt = refiner._build_generation_prompt(
                    profile,
                    "red fox mossy forest",
                    "watercolor style",
                    430,
                ).lower()
                for leaked_term in (
                    "copper-haired archivist",
                    "blue crystal",
                    "narrow wooden workshop",
                    "rain-streaked window",
                ):
                    self.assertNotIn(leaked_term, prompt)

    def test_anchor_is_preserved_while_positive_is_subject_first(self):
        positive, _negative, _report, base, foreground, background = self.result()
        self.assertTrue(base.startswith("mgrtt style,"))
        self.assertTrue(positive.startswith(foreground + "\n\n"))
        self.assertLess(positive.index(foreground), positive.index(background))
        self.assertLess(positive.index(background), positive.index(base))
        self.assertEqual(len(positive.split("\n\n")), 2)

    def test_short_valid_sections_are_not_padded_to_target_minimums(self):
        positive, _negative, _report, base, foreground, background = self.result()
        self.assertLessEqual(refiner._word_count(base), refiner.KREA2_BASE_WORD_RANGE[1])
        self.assertLessEqual(refiner._word_count(foreground), refiner.KREA2_FOREGROUND_WORD_RANGE[1])
        self.assertLessEqual(refiner._word_count(background), refiner.KREA2_BACKGROUND_WORD_RANGE[1])
        self.assertLess(refiner._word_count(positive), refiner.KREA2_POSITIVE_WORD_RANGE[0])

    def test_controls_and_field_labels_are_removed_but_visible_text_survives(self):
        positive = self.result()[0]
        lowered = positive.lower()
        for forbidden in ("score_9", "masterpiece", "style_cluster_", "rating_", "foreground_prompt", "background_prompt"):
            self.assertNotIn(forbidden, lowered)
        self.assertIn('"ARCHIVE_7"', positive)
        self.assertIn("three clear glass bottles", positive)
        self.assertIn("behind the table", positive)
        self.assertIn("to her left", positive)

    def test_meta_openings_are_removed_without_damaging_quoted_text(self):
        values = self.values()
        values["foreground_prompt"] = (
            'The image shows one red fox holding a metal sign that reads "The image shows OPEN". '
            "Its tail points left while both paws touch the brushed steel surface."
        )
        positive = refiner._postprocess_result(
            values,
            "krea2",
            "red fox metal sign",
            "",
            430,
        )[0]
        self.assertTrue(positive.startswith("One red fox"))
        self.assertIn('"The image shows OPEN"', positive)

    def test_negative_baseline_is_added(self):
        negative = self.result()[1]
        for term in ("bad anatomy", "bad hands", "duplicate subject", "distorted geometry", "unreadable text", "watermark"):
            self.assertIn(term, negative)

    def test_local_fallback_remains_krea2_specific(self):
        positive, negative, report, base, foreground, background = refiner._local_fallback_result(
            "krea2",
            "one red fox snow pine trees wooden sign",
            "soft watercolor style",
            430,
            "invalid response",
        )
        self.assertTrue(positive.startswith(foreground))
        self.assertEqual(base, "soft watercolor style")
        self.assertEqual(positive, refiner._join_positive_parts("krea2", base, foreground, background))
        self.assertIn("distorted geometry", negative)
        self.assertIn("stage=local", report.lower())
        self.assertIn("fallback_mode=adaptive", report.lower())
        self.assertNotIn("score_", positive.lower())
        self.assertLess(refiner._word_count(positive), refiner.KREA2_POSITIVE_WORD_RANGE[0])

    def test_node_interface_keeps_existing_outputs_first(self):
        self.assertEqual(refiner.OnekoOllamaPromptRefiner.RETURN_NAMES[:6], refiner.OUTPUT_KEYS)
        self.assertEqual(refiner.OnekoOllamaPromptRefiner.RETURN_NAMES[6:], ("plan_json", "review_json"))
        self.assertEqual(len(refiner.OnekoOllamaPromptRefiner.RETURN_TYPES), 8)


if __name__ == "__main__":
    unittest.main()
