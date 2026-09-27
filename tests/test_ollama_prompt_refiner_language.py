import json
import sys
import unittest
from pathlib import Path
from unittest import mock


from tests.paths import COMFY_ROOT, ONEKO_ROOT
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

from Oneko_ComfyUI_Nodes.nodes import ollama_prompt_refiner as refiner  # noqa: E402


def language_response(
    word_salad,
    *,
    left="",
    right="",
    top="",
    bottom="",
):
    return json.dumps(
        {
            "word_salad": word_salad,
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom,
        },
        ensure_ascii=False,
    )


class LanguageNormalizationTests(unittest.TestCase):
    def test_english_detection_forwards_original_text_byte_for_byte(self):
        source = refiner._language_source_values("  red dragon score_9  ", " left tower ", "", "", "")
        response = json.loads(
            language_response(
                "red dragon score_9",
                left="left tower",
            )
        )

        values, detected, translated = refiner._validate_language_result(response, source)

        self.assertEqual(values, source)
        self.assertEqual(detected, "english")
        self.assertFalse(translated)

    def test_german_and_spatial_text_are_translated_without_moving_fields(self):
        source = refiner._language_source_values(
            'rote Drachin <lora:hero:0.8> score_9 vor einem Schild "Grüße"',
            "ein alter Turm",
            "blue crystal",
            "",
            "nasses Gras",
        )
        response = json.loads(
            language_response(
                'red female dragon <lora:hero:0.8> score_9 in front of a sign reading "Grüße"',
                left="an old tower",
                right="blue crystal",
                bottom="wet grass",
            )
        )

        values, detected, translated = refiner._validate_language_result(response, source)

        self.assertEqual(values["left"], "an old tower")
        self.assertEqual(values["right"], "blue crystal")
        self.assertEqual(values["top"], "")
        self.assertEqual(values["bottom"], "wet grass")
        self.assertIn("<lora:hero:0.8>", values["word_salad"])
        self.assertIn('"Grüße"', values["word_salad"])
        self.assertEqual(detected, "mixed")
        self.assertTrue(translated)

    def test_translation_rejects_lost_protected_tokens_and_changed_empty_fields(self):
        source = refiner._language_source_values('rote Drachin score_9 "Grüße"', "", "", "", "")
        lost_token = json.loads(language_response("red female dragon"))
        with self.assertRaisesRegex(ValueError, "changed protected tokens"):
            refiner._validate_language_result(lost_token, source)

        filled_empty = json.loads(language_response('red female dragon score_9 "Grüße"', left="tower"))
        with self.assertRaisesRegex(ValueError, "changed whether left is empty"):
            refiner._validate_language_result(filled_empty, source)

    def test_language_request_is_deterministic_and_uses_the_structured_schema(self):
        response = language_response(
            "red dragon",
            left="old tower",
        )
        with mock.patch.object(refiner, "_request_ollama", return_value=response) as request:
            values, detected, translated = refiner._prepare_language_inputs(
                "roter Drache",
                "alter Turm",
                "",
                "",
                "",
                "http://127.0.0.1:11434",
                "test-model",
                17,
                30,
                8192,
            )

        self.assertEqual(values["word_salad"], "red dragon")
        self.assertEqual(values["left"], "old tower")
        self.assertEqual((detected, translated), ("german", True))
        self.assertEqual(request.call_count, 1)
        args, kwargs = request.call_args
        self.assertEqual(args[3:8], (17, 0.0, 1.0, 30, 8192))
        self.assertIs(kwargs["output_schema"], refiner.LANGUAGE_SCHEMA)
        self.assertIs(kwargs["system_instructions"], refiner.LANGUAGE_SYSTEM_INSTRUCTIONS)
        self.assertEqual(kwargs["num_predict"], 256)
        self.assertFalse(kwargs["reasoning"])

    def test_invalid_language_json_is_repaired_once(self):
        repaired = language_response("red dragon")
        with mock.patch.object(refiner, "_request_ollama", side_effect=["not json", repaired]) as request:
            values, detected, translated = refiner._prepare_language_inputs(
                "roter Drache",
                "",
                "",
                "",
                "",
                "http://127.0.0.1:11434",
                "test-model",
                9,
                30,
                4096,
            )

        self.assertEqual(values["word_salad"], "red dragon")
        self.assertEqual((detected, translated), ("german", True))
        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args_list[1].args[3], 10)
        self.assertIn("Invalid response:\nnot json", request.call_args_list[1].args[2])

    def test_long_source_gets_room_for_all_fields_on_both_attempts(self):
        source = refiner._language_source_values(
            "A red dragon rests beside an old stone tower. " * 35,
            "Three blue lanterns hang from a wooden gate. " * 10,
        )
        response = json.dumps(source)
        with mock.patch.object(refiner, "_request_ollama", side_effect=["bad", response]) as request:
            values, _, translated = refiner._prepare_language_inputs(
                *source.values(), "url", "model", 9, 30, 16384
            )

        self.assertEqual(values, source)
        self.assertFalse(translated)
        budgets = [call.kwargs["num_predict"] for call in request.call_args_list]
        self.assertGreater(budgets[0], len(response.split()))
        self.assertEqual(budgets[0], budgets[1])

    def test_changed_quoted_text_triggers_one_protected_token_repair(self):
        source = 'roter Drache vor einem Schild "Gr??e"'
        invalid = language_response('red dragon in front of a sign "Greetings"')
        repaired = language_response('red dragon in front of a sign "Gr??e"')
        with mock.patch.object(refiner, "_request_ollama", side_effect=[invalid, repaired]) as request:
            values, detected, translated = refiner._prepare_language_inputs(
                source, "", "", "", "", "url", "model", 3, 30, 4096
            )

        self.assertEqual(values["word_salad"], 'red dragon in front of a sign "Gr??e"')
        self.assertEqual((detected, translated), ("german", True))
        self.assertEqual(request.call_count, 2)
        self.assertIn("changed protected tokens", request.call_args_list[1].args[2])

    def test_second_invalid_response_and_transport_failure_are_actionable(self):
        with mock.patch.object(refiner, "_request_ollama", side_effect=["bad", "still bad"]):
            with self.assertRaisesRegex(RuntimeError, "invalid JSON twice"):
                refiner._prepare_language_inputs(
                    "roter Drache", "", "", "", "", "url", "model", 0, 30, 4096
                )

        with mock.patch.object(refiner, "_request_ollama", side_effect=RuntimeError("offline")) as request:
            with self.assertRaisesRegex(RuntimeError, "automatic language stage failed.*offline"):
                refiner._prepare_language_inputs(
                    "roter Drache", "", "", "", "", "url", "model", 0, 30, 4096
                )
        self.assertEqual(request.call_count, 1)

    def test_refiner_uses_canonical_inputs_and_never_changes_style_anchor(self):
        canonical = refiner._language_source_values(
            "red dragon",
            "old tower",
            "blue crystal",
            "golden clouds",
            "wet grass",
        )
        result = ("positive", "negative", "report", "base", "foreground", "background", "{}", "{}")
        node = refiner.OnekoOllamaPromptRefiner()
        with mock.patch.object(
            refiner,
            "_prepare_language_inputs",
            return_value=(canonical, "mixed", True),
        ), mock.patch.object(node, "_refine_canonical", return_value=result) as refine_canonical:
            output = node._refine(
                "roter Drache",
                "url",
                "model",
                "krea2",
                5,
                0.4,
                0.9,
                430,
                30,
                8192,
                "<lora:fixed:0.8> Künstlername",
                "alter Turm",
                "blue crystal",
                "goldene Wolken",
                "nasses Gras",
            )

        args = refine_canonical.call_args.args
        self.assertEqual(args[0], "red dragon")
        self.assertEqual(args[10], "<lora:fixed:0.8> Künstlername")
        self.assertEqual(args[11:15], ("old tower", "blue crystal", "golden clouds", "wet grass"))
        self.assertEqual(output[6:], ("{}", "{}"))
        self.assertIn("translated source text to English", output[2])

    def test_unicode_token_helpers_keep_umlauts(self):
        self.assertEqual(refiner._curated_terms("Mädchen Ölgemälde"), ["Mädchen", "Ölgemälde"])
        self.assertEqual(refiner._prompt_token("Mädchen"), "mädchen")
        self.assertEqual(refiner._concept_tokens("Mädchen Ölgemälde"), ["mädchen", "ölgemälde"])

    def test_language_failure_still_unloads_the_model(self):
        node = refiner.OnekoOllamaPromptRefiner()
        with mock.patch.object(
            refiner,
            "_prepare_language_inputs",
            side_effect=RuntimeError("translation failed"),
        ), mock.patch.object(refiner, "_unload_after_run") as unload:
            with self.assertRaisesRegex(RuntimeError, "translation failed"):
                node.refine(
                    "roter Drache",
                    "url",
                    "model",
                    "krea2",
                    0,
                    0.4,
                    0.9,
                    430,
                    30,
                )
        unload.assert_called_once_with("url", "model", 30, True)

    def test_tooltips_document_automatic_german_input(self):
        inputs = refiner.OnekoOllamaPromptRefiner.INPUT_TYPES()
        self.assertIn("German", inputs["required"]["word_salad"][1]["tooltip"])
        self.assertIn("German", inputs["optional"]["left"][1]["tooltip"])
        self.assertIn("translates German", refiner.OnekoOllamaPromptRefiner.DESCRIPTION)


if __name__ == "__main__":
    unittest.main()
