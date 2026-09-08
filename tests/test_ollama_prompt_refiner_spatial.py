import json
import sys
import unittest
from pathlib import Path
from unittest import mock


from tests.paths import COMFY_ROOT, ONEKO_ROOT
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

from Oneko_ComfyUI_Nodes.nodes import ollama_prompt_refiner as refiner


def passthrough_language_inputs(word_salad, left, right, top, bottom, *_args):
    return refiner._language_source_values(word_salad, left, right, top, bottom), "english", False


SPATIAL_VALUES = {
    "left": "a red fox beside a mossy stone",
    "right": "a blue glass tower reflected in water",
    "top": "golden clouds crossing the moon",
    "bottom": "wild grass and scattered copper tools",
}


class SpatialPromptInputTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(
            refiner,
            "_prepare_language_inputs",
            side_effect=passthrough_language_inputs,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def spatial_result(self, foreground="A central subject stands in clear light.", background="A detailed room surrounds the subject."):
        base = "A cinematic image uses balanced composition and natural color."
        positive = refiner._join_positive_parts("krea2", base, foreground, background)
        return positive, "blurry", "Built a spatial scene.", base, foreground, background

    def ollama_response_with_integrated_left(self):
        return json.dumps(
            {
                "base_prompt": "A cinematic medium shot uses clear natural light and balanced color.",
                "foreground_prompt": "On the left, a red fox rests beside a mossy stone with a readable pose and detailed fur.",
                "background_prompt": "A narrow workshop has wooden shelves, warm lamps, and distant city rooftops.",
                "negative": "blurry, bad anatomy",
                "report": "Built a coherent spatial scene.",
            }
        )

    def test_optional_inputs_are_multiline_connectable_strings_and_outputs_stay_stable(self):
        optional = refiner.OnekoOllamaPromptRefiner.INPUT_TYPES()["optional"]
        for name in refiner.SPATIAL_INPUT_KEYS:
            value_type, options = optional[name]
            self.assertEqual(value_type, "STRING")
            self.assertTrue(options["multiline"])
            self.assertTrue(options["dynamicPrompts"])
            self.assertTrue(options["defaultInput"])
        self.assertEqual(
            refiner.OnekoOllamaPromptRefiner.RETURN_NAMES[:6],
            ("positive", "negative", "report", "base_prompt", "foreground_prompt", "background_prompt"),
        )

    def test_supported_profiles_receive_all_labeled_regions(self):
        for profile in refiner.SPATIAL_TARGET_PROFILES:
            with self.subTest(profile=profile):
                prompt = refiner._build_generation_prompt(
                    profile,
                    "global cinematic motif",
                    "",
                    430,
                    **SPATIAL_VALUES,
                )
                labels = (
                    (
                        "Left placement in the shared frame",
                        "Right placement in the shared frame",
                        "Upper placement in the shared frame",
                        "Lower placement in the shared frame",
                    )
                    if profile == "anima"
                    else ("Left side", "Right side", "Top area", "Bottom area")
                )
                for label, value in zip(labels, SPATIAL_VALUES.values()):
                    self.assertIn(f"{label}: {value}", prompt)
                self.assertIn("creative composition guidance rather than rigid geometry", prompt)
                self.assertIn("global guidance for the entire image", prompt)

    def test_pony_v7_keeps_regions_out_of_control_tags(self):
        prompt = refiner._build_generation_prompt("pony_v7", "forest", "", 430, **SPATIAL_VALUES)
        self.assertIn("only in the natural caption sections", prompt)
        self.assertIn("not as control or Danbooru tags", prompt)

    def test_anima_spatial_context_requires_one_coherent_camera_view(self):
        context = refiner._spatial_context("anima", **SPATIAL_VALUES)
        for phrase in (
            "one continuous full-frame image",
            "one camera view",
            "Describe the main figure exactly once",
            "Never turn the placements into panels",
            "separate close-up and full-body views",
            "one identity, outfit, body scale, pose, and silhouette",
        ):
            self.assertIn(phrase, context)

    def test_anima_generation_repair_and_retry_keep_coherence_rules(self):
        prompts = (
            refiner._build_generation_prompt("anima", "forest", "", 430, **SPATIAL_VALUES),
            refiner._build_repair_prompt("not json", "anima", **SPATIAL_VALUES),
            refiner._build_minimal_retry_prompt("anima", "forest", "", 430, **SPATIAL_VALUES),
        )
        for prompt in prompts:
            self.assertIn("one continuous full-frame image", prompt)
            self.assertIn("Describe the main figure exactly once", prompt)

    def test_anima_safety_sentence_reinforces_one_continuous_composition(self):
        text = refiner._spatial_fallback_text(
            "anima",
            right="one dark-haired cat woman holding a lantern",
            bottom="mushrooms and fallen logs",
        )
        self.assertEqual(
            text,
            "Within the same continuous composition on the right, one dark-haired cat woman holding a lantern. "
            "Within the same continuous composition along the lower area, mushrooms and fallen logs.",
        )

    def test_anima_existing_subject_gets_positioned_without_repeating_details(self):
        text = refiner._spatial_fallback_text(
            "anima",
            right="one dark-haired cat woman holding a glowing lantern",
            existing_text=(
                "One dark-haired cat woman holds a glowing lantern beside her black lace dress. "
                "A shadowy forest surrounds her."
            ),
        )
        self.assertEqual(
            text,
            "The same main figure occupies the right side of the single continuous composition.",
        )
        self.assertNotIn("dark-haired", text)
        self.assertNotIn("lantern", text)

    def test_anima_spatial_result_adds_multi_view_negatives(self):
        result = refiner._apply_spatial_result(
            self.spatial_result(),
            "anima",
            right="one dark-haired cat woman holding a lantern",
        )
        for term in refiner.ANIMA_SPATIAL_NEGATIVE_TERMS:
            self.assertIn(term, result[1])

    def test_anima_local_fallback_prioritizes_regional_subject(self):
        result = refiner._local_fallback_result(
            "anima",
            "logo mark style dynamic composition strong composition delicate details year 2000 score_8",
            "anime illustration",
            430,
            "invalid json",
            right="anthro cat woman with long dark hair black lace dress holding a glowing lantern",
            bottom="shadowy forest mushrooms fallen logs",
        )
        foreground = result[4].lower()
        self.assertIn("anthro", foreground)
        self.assertIn("cat", foreground)
        self.assertIn("woman", foreground)
        self.assertLess(foreground.index("anthro"), foreground.find("logo") if "logo" in foreground else len(foreground))
        self.assertTrue(
            result[5].startswith(
                "The same main figure occupies the right side of the single continuous composition."
            )
        )
        self.assertEqual(result[0].lower().count("anthro"), 1)

    def test_tag_profiles_ignore_spatial_data(self):
        for profile in ("pony_v6", "illustrious"):
            with self.subTest(profile=profile):
                prompt = refiner._build_generation_prompt(profile, "forest", "", 430, **SPATIAL_VALUES)
                self.assertNotIn("Spatial composition hints", prompt)
                for value in SPATIAL_VALUES.values():
                    self.assertNotIn(value, prompt)

    def test_empty_regions_do_not_create_placeholders(self):
        context = refiner._spatial_context("krea2", left=SPATIAL_VALUES["left"])
        self.assertIn("Left side:", context)
        for absent_label in ("Right side:", "Top area:", "Bottom area:"):
            self.assertNotIn(absent_label, context)

    def test_repair_and_minimal_retry_retain_spatial_context(self):
        repair = refiner._build_repair_prompt("not json", "anima", **SPATIAL_VALUES)
        minimal = refiner._build_minimal_retry_prompt(
            "wan2_2_video", "global motion", "", 430, **SPATIAL_VALUES
        )
        for prompt in (repair, minimal):
            for value in SPATIAL_VALUES.values():
                self.assertIn(value, prompt)

    def test_local_fallback_turns_only_nonempty_regions_into_sentences(self):
        result = refiner._local_fallback_result(
            "krea2",
            "",
            "",
            430,
            "three invalid responses",
            left="a red fox",
            top="golden clouds",
        )
        positive, _negative, _report, _base, _foreground, background = result
        self.assertIn("On the left, a red fox.", background)
        self.assertIn("Across the top, golden clouds.", background)
        self.assertTrue(background.startswith("On the left, a red fox. Across the top, golden clouds."))
        self.assertIn("On the left, a red fox.", positive)
        self.assertNotIn("On the right", background)
        self.assertNotIn("Along the bottom", background)

    def test_spatial_only_natural_profile_reaches_initial_repair_retry_and_fallback(self):
        node = refiner.OnekoOllamaPromptRefiner()
        with mock.patch.object(refiner, "_request_ollama", side_effect=("invalid", "still invalid", "again invalid")) as request:
            result = node.refine(
                "",
                "http://127.0.0.1:11434",
                "test-model",
                "krea2",
                7,
                0.4,
                0.9,
                430,
                30,
                8192,
                "",
                left="a red fox",
                bottom="wet cobblestones",
                unload_after_run=False,
            )
        self.assertEqual(request.call_count, 3)
        for call in request.call_args_list:
            sent_prompt = call.args[2]
            self.assertIn("a red fox", sent_prompt)
            self.assertIn("wet cobblestones", sent_prompt)
        self.assertIn("On the left, a red fox.", result[0])
        self.assertIn("Along the bottom, wet cobblestones.", result[0])

    def test_valid_ollama_response_cannot_drop_connected_regions(self):
        response = """{
            "base_prompt": "A cinematic medium shot uses clear natural light and balanced color.",
            "foreground_prompt": "One copper-haired archivist studies a blue crystal beside a brass machine, with detailed clothing, careful hands, and clear material textures.",
            "background_prompt": "A narrow workshop has wooden shelves, rain-streaked windows, warm lamps, and distant city rooftops.",
            "negative": "blurry, bad anatomy",
            "report": "Built a coherent workshop scene."
        }"""
        node = refiner.OnekoOllamaPromptRefiner()
        with mock.patch.object(refiner, "_request_ollama", return_value=response) as request:
            result = node.refine(
                "global archive idea",
                "http://127.0.0.1:11434",
                "test-model",
                "krea2",
                7,
                0.4,
                0.9,
                430,
                30,
                8192,
                "anime style furry anthro",
                left="boy nude presenting penis cum",
                bottom="furry latex thighhighs vaginal sex nipples lying presenting anus",
                unload_after_run=False,
            )
        self.assertEqual(request.call_count, 1)
        self.assertIn("On the left, boy nude presenting penis cum.", result[5])
        self.assertIn(
            "Along the bottom, furry latex thighhighs vaginal sex nipples lying presenting anus.",
            result[5],
        )
        self.assertIn("On the left, boy nude presenting penis cum.", result[0])

    def test_integrated_direction_and_content_are_not_duplicated(self):
        background = "On the left, a red fox rests beside a mossy stone. Warm windows illuminate the room."
        result = refiner._apply_spatial_result(
            self.spatial_result(background=background),
            "krea2",
            left="a red fox beside a mossy stone",
        )
        self.assertEqual(result[5], background)
        self.assertEqual(result[5].lower().count("red fox"), 1)

    def test_content_match_threshold_scales_up_to_three_words(self):
        self.assertTrue(refiner._spatial_region_is_integrated("left", "fox", "On the left, a fox waits."))
        self.assertFalse(
            refiner._spatial_region_is_integrated(
                "left",
                "red fox",
                "On the left, a red shape waits.",
            )
        )
        self.assertTrue(
            refiner._spatial_region_is_integrated(
                "left",
                "red fox mossy stone",
                "On the left, a red fox waits beside a stone.",
            )
        )

    def test_right_hand_is_not_mistaken_for_right_side(self):
        self.assertFalse(
            refiner._spatial_region_is_integrated(
                "right",
                "blue glove",
                "His right hand wears a blue glove.",
            )
        )

    def test_content_without_direction_triggers_short_safety_sentence(self):
        background = "A red fox rests beside a mossy stone near warm windows."
        result = refiner._apply_spatial_result(
            self.spatial_result(background=background),
            "krea2",
            left="a red fox beside a mossy stone",
        )
        self.assertTrue(result[5].startswith("On the left, a red fox beside a mossy stone."))

    def test_direction_without_matching_content_triggers_short_safety_sentence(self):
        background = "On the left, a blue glass tower reflects the evening light."
        result = refiner._apply_spatial_result(
            self.spatial_result(background=background),
            "krea2",
            left="a red fox beside a mossy stone",
        )
        self.assertTrue(result[5].startswith("On the left, a red fox beside a mossy stone."))

    def test_only_missing_regions_are_prepended_in_fixed_order(self):
        foreground = "On the left, a red fox waits beside a mossy stone."
        background = "Across the top, golden clouds cross the pale moon above a quiet valley."
        result = refiner._apply_spatial_result(
            self.spatial_result(foreground=foreground, background=background),
            "krea2",
            **SPATIAL_VALUES,
        )
        expected = (
            "On the right, a blue glass tower reflected in water. "
            "Along the bottom, wild grass and scattered copper tools."
        )
        self.assertTrue(result[5].startswith(expected))
        self.assertEqual(result[0].lower().count("on the left"), 1)
        self.assertEqual(result[0].lower().count("across the top"), 1)

    def test_repair_and_minimal_success_use_the_same_missing_region_logic(self):
        node = refiner.OnekoOllamaPromptRefiner()
        common = dict(
            word_salad="global scene idea",
            ollama_url="http://127.0.0.1:11434",
            ollama_model="test-model",
            target_profile="krea2",
            seed=7,
            temperature=0.4,
            top_p=0.9,
            style_cluster=430,
            timeout_seconds=30,
            context_length=8192,
            style_anchor="",
            left="a red fox beside a mossy stone",
            bottom="wet cobblestones",
            unload_after_run=False,
        )
        for responses, expected_calls in (
            (("invalid", self.ollama_response_with_integrated_left()), 2),
            (("invalid", "still invalid", self.ollama_response_with_integrated_left()), 3),
        ):
            with self.subTest(expected_calls=expected_calls):
                with mock.patch.object(refiner, "_request_ollama", side_effect=responses) as request:
                    result = node.refine(**common)
                self.assertEqual(request.call_count, expected_calls)
                self.assertEqual(result[0].lower().count("on the left"), 1)
                self.assertTrue(result[5].startswith("Along the bottom, wet cobblestones."))

    def test_pony_v7_safety_sentence_stays_in_natural_background_caption(self):
        base = "score_9, rating_explicit, style_cluster_430\n\n# stylistic description\nDigital anime illustration."
        foreground = "A central character stands beneath soft light."
        background = "A detailed room surrounds the character."
        positive = refiner._join_positive_parts("pony_v7", base, foreground, background)
        result = refiner._apply_spatial_result(
            (positive, "blurry", "Built Pony v7.", base, foreground, background),
            "pony_v7",
            left="a red fox beside a mossy stone",
        )
        self.assertEqual(result[3], base)
        self.assertTrue(result[5].startswith("On the left, a red fox beside a mossy stone."))
        self.assertNotIn("red fox", result[3].lower())

    def test_region_text_normalizes_multiline_trailing_punctuation(self):
        text = refiner._spatial_fallback_text(
            "krea2",
            left="  a red fox\n beside a stone...  ",
        )
        self.assertEqual(text, "On the left, a red fox beside a stone.")

    def test_unsupported_profile_never_receives_safety_sentences(self):
        original = self.spatial_result()
        result = refiner._apply_spatial_result(original, "pony_v6", **SPATIAL_VALUES)
        self.assertEqual(result, original)

    def test_empty_input_and_spatial_only_tag_profiles_fail_clearly(self):
        node = refiner.OnekoOllamaPromptRefiner()
        common = (
            "",
            "http://127.0.0.1:11434",
            "test-model",
            "pony_v6",
            0,
            0.4,
            0.9,
            430,
            30,
        )
        with self.assertRaisesRegex(RuntimeError, "must contain text"):
            node.refine(*common, unload_after_run=False)
        with self.assertRaisesRegex(RuntimeError, "spatial-only input requires a natural prompt profile"):
            node.refine(*common, left="a red fox", unload_after_run=False)

    def test_cache_hash_changes_for_each_spatial_field(self):
        common = dict(
            word_salad="global idea",
            ollama_url="http://127.0.0.1:11434",
            ollama_model="test-model",
            target_profile="krea2",
            seed=0,
            temperature=0.4,
            top_p=0.9,
            style_cluster=430,
            timeout_seconds=30,
            context_length=8192,
            style_anchor="",
        )
        baseline = refiner.OnekoOllamaPromptRefiner.IS_CHANGED(**common)
        hashes = {
            name: refiner.OnekoOllamaPromptRefiner.IS_CHANGED(**common, **{name: "regional idea"})
            for name in refiner.SPATIAL_INPUT_KEYS
        }
        self.assertTrue(all(value != baseline for value in hashes.values()))
        self.assertEqual(len(set(hashes.values())), len(refiner.SPATIAL_INPUT_KEYS))


if __name__ == "__main__":
    unittest.main()
