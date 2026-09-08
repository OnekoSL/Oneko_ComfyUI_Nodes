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


def response_json(profile="anima"):
    negative = "" if profile == "z_image" else "bad anatomy"
    return json.dumps(
        {
            "base_prompt": "Soft painted light shapes a balanced composition.",
            "foreground_prompt": "A crimson dragon coils around a crystal tower.",
            "background_prompt": "Moonlit clouds drift above a silent mountain valley.",
            "negative": negative,
            "report": "Kept the unusual visual subject.",
        }
    )


def refine_kwargs(profile="anima", fallback_mode="adaptive", **overrides):
    values = {
        "word_salad": "crimson dragon crystal tower moonlit valley",
        "ollama_url": "http://127.0.0.1:11434",
        "ollama_model": "test-model",
        "target_profile": profile,
        "seed": 7,
        "temperature": 0.4,
        "top_p": 0.9,
        "style_cluster": 430,
        "timeout_seconds": 30,
        "context_length": 8192,
        "style_anchor": "",
        "left": "",
        "right": "",
        "top": "",
        "bottom": "",
        "fallback_mode": fallback_mode,
        "unload_after_run": False,
    }
    values.update(overrides)
    return values


class NaturalFallbackModeTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(
            refiner,
            "_prepare_language_inputs",
            side_effect=passthrough_language_inputs,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_interface_exposes_exactly_three_modes_with_adaptive_default(self):
        optional = refiner.OnekoOllamaPromptRefiner.INPUT_TYPES()["optional"]
        choices, options = optional["fallback_mode"]
        self.assertEqual(choices, ("adaptive", "strict", "continue"))
        self.assertEqual(options["default"], "adaptive")
        self.assertEqual(len(refiner.OnekoOllamaPromptRefiner.RETURN_TYPES), 8)

    def test_adaptive_preserves_short_single_sentence_for_every_natural_profile(self):
        for profile in refiner.NATURAL_FALLBACK_PROFILES:
            with self.subTest(profile=profile):
                values = json.loads(response_json(profile))
                result = refiner._postprocess_result(
                    values,
                    profile,
                    "crimson dragon crystal tower moonlit valley",
                    "",
                    fallback_mode="adaptive",
                )
                self.assertEqual(
                    result[4].rstrip("."),
                    "A crimson dragon coils around a crystal tower",
                )
                self.assertEqual(
                    result[5].rstrip("."),
                    "Moonlit clouds drift above a silent mountain valley",
                )

    def test_adaptive_local_rescue_has_no_removed_standard_foreground(self):
        removed_boilerplate = (
            "The main figure reflects",
            "The face has carefully shaped features",
            "The camera keeps the figure dominant",
            "appears as a real visual presence",
        )
        for profile in refiner.NATURAL_FALLBACK_PROFILES:
            with self.subTest(profile=profile):
                result = refiner._local_fallback_result(
                    profile,
                    "crimson dragon crystal tower moonlit valley",
                    "",
                    430,
                    "three invalid responses",
                    fallback_mode="adaptive",
                )
                self.assertTrue(result[4])
                for phrase in removed_boilerplate:
                    self.assertNotIn(phrase, result[4])

    def test_strict_tries_all_three_responses_then_raises(self):
        node = refiner.OnekoOllamaPromptRefiner()
        with mock.patch.object(refiner, "_request_ollama", side_effect=("invalid", "still invalid", "again invalid")) as request:
            with self.assertRaisesRegex(RuntimeError, "strict mode rejected all three responses"):
                node.refine(**refine_kwargs(fallback_mode="strict"))
        self.assertEqual(request.call_count, 3)

    def test_strict_retries_when_spatial_content_was_omitted(self):
        node = refiner.OnekoOllamaPromptRefiner()
        response = response_json("anima")
        with mock.patch.object(refiner, "_request_ollama", side_effect=(response, response, response)) as request:
            with self.assertRaisesRegex(RuntimeError, "omitted one or more connected spatial inputs"):
                node.refine(
                    **refine_kwargs(
                        fallback_mode="strict",
                        right="a brass telescope beside the main figure",
                    )
                )
        self.assertEqual(request.call_count, 3)

    def test_continue_merges_partial_json_fields_across_attempts(self):
        responses = (
            json.dumps({"foreground_prompt": "A crimson dragon coils around a crystal tower.", "report": "partial"}),
            json.dumps({"background_prompt": "Moonlit clouds drift above a silent mountain valley."}),
            json.dumps({"base_prompt": "Painterly fantasy lighting shapes the composition."}),
        )
        node = refiner.OnekoOllamaPromptRefiner()
        with mock.patch.object(refiner, "_request_ollama", side_effect=responses) as request:
            result = node.refine(**refine_kwargs(fallback_mode="continue"))
        self.assertEqual(request.call_count, 3)
        self.assertIn("crimson dragon", result[4].lower())
        self.assertIn("moonlit clouds", result[5].lower())
        self.assertIn("painterly fantasy", result[3].lower())
        self.assertIn("initial", result[2])
        self.assertIn("repair", result[2])
        self.assertIn("minimal", result[2])

    def test_continue_uses_non_json_raw_prose(self):
        raw = "A silver wyvern circles a ruined observatory beneath green storm clouds."
        node = refiner.OnekoOllamaPromptRefiner()
        with mock.patch.object(refiner, "_request_ollama", side_effect=(raw, raw, raw)):
            result = node.refine(**refine_kwargs(fallback_mode="continue"))
        self.assertIn("silver wyvern circles a ruined observatory", result[4].lower())
        self.assertNotIn("The main figure reflects", result[0])

    def test_continue_ignores_transport_error_and_passes_connected_inputs(self):
        node = refiner.OnekoOllamaPromptRefiner()
        with mock.patch.object(refiner, "_request_ollama", side_effect=RuntimeError("connection refused")) as request:
            result = node.refine(
                **refine_kwargs(
                    fallback_mode="continue",
                    word_salad="silver wyvern ruined observatory",
                    style_anchor="watercolor illustration",
                    right="green storm clouds above a brass telescope",
                )
            )
        self.assertEqual(request.call_count, 1)
        self.assertTrue(result[3].startswith("watercolor illustration"))
        self.assertIn("silver wyvern", result[4].lower())
        self.assertIn("green storm clouds", result[5].lower())
        self.assertIn("connection refused", result[2])

    def test_continue_keeps_z_image_negative_empty(self):
        node = refiner.OnekoOllamaPromptRefiner()
        with mock.patch.object(refiner, "_request_ollama", side_effect=RuntimeError("offline")):
            result = node.refine(**refine_kwargs(profile="z_image", fallback_mode="continue"))
        self.assertEqual(result[1], "")

    def test_report_names_initial_stage_mode_and_preserved_sections(self):
        node = refiner.OnekoOllamaPromptRefiner()
        with mock.patch.object(refiner, "_request_ollama", return_value=response_json("anima")):
            result = node.refine(**refine_kwargs(fallback_mode="adaptive"))
        self.assertIn("stage=initial", result[2])
        self.assertIn("fallback_mode=adaptive", result[2])
        self.assertIn("foreground=preserved", result[2])
        self.assertIn("background=preserved", result[2])

    def test_cache_hash_tracks_mode_only_for_natural_profiles(self):
        anima_adaptive = refiner.OnekoOllamaPromptRefiner.IS_CHANGED(**refine_kwargs("anima", "adaptive"))
        anima_strict = refiner.OnekoOllamaPromptRefiner.IS_CHANGED(**refine_kwargs("anima", "strict"))
        pony_strict = refiner.OnekoOllamaPromptRefiner.IS_CHANGED(**refine_kwargs("pony_v6", "strict"))
        pony_continue = refiner.OnekoOllamaPromptRefiner.IS_CHANGED(**refine_kwargs("pony_v6", "continue"))
        self.assertNotEqual(anima_adaptive, anima_strict)
        self.assertEqual(pony_strict, pony_continue)


if __name__ == "__main__":
    unittest.main()
