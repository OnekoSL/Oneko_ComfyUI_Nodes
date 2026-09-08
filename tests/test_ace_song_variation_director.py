import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


from tests.paths import COMFY_ROOT, ONEKO_ROOT
CUSTOM_NODE_ROOT = ONEKO_ROOT
for import_path in (COMFY_ROOT, CUSTOM_NODE_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from Oneko_ComfyUI_Nodes.nodes import ace_song_variation_director as director


SOURCE_TAGS = "dark Irish pub folk, tenor, acoustic guitar"
SOURCE_LYRICS = """[Strophe 1]
In Notmark steht der Baron schwer.

[Refrain]
Man nennt ihn Warzensau im Land.

[Verse 2]
Die Nacht zieht über Moor und Wehr."""


def _valid_response(additional_sections=None):
    return {
        "global_arrangement": "Begin sparse, intensify each refrain, and finish with a rough communal climax.",
        "source_sections": [
            {
                "id": "S01",
                "direction": "restrained tenor over sparse fingerpicked guitar",
                "lyrics": "Zu Notmark zählt der Baron Gold und schweigt.",
            },
            {
                "id": "S02",
                "direction": "faster communal response with stomping pulse",
                "lyrics": "Hebt an und ruft: Die Warzensau wird fallen!",
            },
            {
                "id": "S03",
                "direction": "half-time tension with low vocal register",
                "lyrics": "Das Moor erwacht, bis jeder Schatten steigt.",
            },
        ],
        "additional_sections": list(additional_sections or []),
        "report": "Rewrote the verses and increased contrast while preserving the narrative anchors.",
    }


def _kwargs(**overrides):
    values = {
        "tags": SOURCE_TAGS,
        "lyrics": SOURCE_LYRICS,
        "must_keep": "Notmark\nBaron\nWarzensau",
        "ollama_url": director.DEFAULT_OLLAMA_URL,
        "ollama_model": director.DEFAULT_DIRECTOR_MODEL,
        "seed": 42,
        "variation_strength": 0.65,
        "energy_variation": 0.75,
        "rhythm_variation": 0.55,
        "instrument_rotation": 0.75,
        "vocal_variation": 0.55,
        "harmonic_variation": 0.25,
        "transition_strength": 0.45,
        "max_new_sections": 1,
        "lyrics_language": "de",
        "temperature": 0.65,
        "top_p": 0.9,
        "timeout_seconds": 180,
        "context_length": "8192",
        "fallback_mode": "passthrough",
        "unload_after_run": True,
    }
    values.update(overrides)
    return values


class AceSongVariationDirectorTests(unittest.TestCase):
    def test_node_registration_and_public_contract(self):
        self.assertIs(
            director.NODE_CLASS_MAPPINGS["OnekoAceSongVariationDirector"],
            director.OnekoAceSongVariationDirector,
        )
        self.assertEqual(
            director.NODE_DISPLAY_NAME_MAPPINGS["OnekoAceSongVariationDirector"],
            "ACE Song Variation Director (Oneko)",
        )
        self.assertEqual(director.OnekoAceSongVariationDirector.CATEGORY, "Oneko/08 Audio/ACE")
        self.assertEqual(
            director.OnekoAceSongVariationDirector.RETURN_NAMES,
            ("tags", "lyrics", "report", "plan_json"),
        )

        with patch.object(director, "_available_ollama_models", return_value=[director.DEFAULT_DIRECTOR_MODEL]):
            inputs = director.OnekoAceSongVariationDirector.INPUT_TYPES()["required"]
        defaults = {
            "variation_strength": 0.65,
            "energy_variation": 0.75,
            "rhythm_variation": 0.55,
            "instrument_rotation": 0.75,
            "vocal_variation": 0.55,
            "harmonic_variation": 0.25,
            "transition_strength": 0.45,
        }
        for name, expected in defaults.items():
            self.assertEqual(inputs[name][1]["default"], expected)
        self.assertEqual(inputs["ollama_model"][1]["default"], director.DEFAULT_DIRECTOR_MODEL)
        self.assertEqual(inputs["max_new_sections"][1]["default"], 1)
        self.assertEqual(inputs["fallback_mode"][1]["default"], "passthrough")
        self.assertTrue(inputs["unload_after_run"][1]["default"])

    def test_parses_german_english_and_unstructured_sections(self):
        sections = director._parse_sections(SOURCE_LYRICS)
        self.assertEqual([item["id"] for item in sections], ["S01", "S02", "S03"])
        self.assertEqual([item["header"] for item in sections], ["Strophe 1", "Refrain", "Verse 2"])

        unstructured = director._parse_sections("Nur eine Zeile\nund eine zweite")
        self.assertEqual(unstructured, [{"id": "S01", "header": "Song", "lyrics": "Nur eine Zeile\nund eine zweite"}])
        self.assertEqual(director._detect_lyrics_language(SOURCE_LYRICS), "de")
        self.assertEqual(director._detect_lyrics_language("The moon rises over an empty road."), "en")

    def test_detects_unstructured_stanzas_and_refrains(self):
        lyrics = """Erste Strophe zieht durchs Land.
Der kalte Wind verweht den Sand.

Hebt eure Krüge, ruft es laut!
Der Baron hat das Land beraubt!

Die zweite Strophe warnt die Stadt.
Weil niemand mehr zu hoffen hat.

Hebt eure Krüge, ruft es laut!
Der Baron hat das Land beraubt!

Hebt eure Krüge, ruft es laut!
Der Baron hat fast ausgespielt!"""
        sections = director._parse_sections(lyrics)
        self.assertEqual(
            [item["header"] for item in sections],
            ["Verse 1", "Chorus", "Verse 2", "Chorus", "Final Chorus"],
        )
        self.assertEqual([item["id"] for item in sections], ["S01", "S02", "S03", "S04", "S05"])

    def test_success_rewrites_lyrics_and_inserts_one_new_section(self):
        extra = {
            "insert_after": "S02",
            "header": "Instrumental Bridge",
            "direction": "brief guitar break with rising drum accents",
            "lyrics": "",
        }
        response = json.dumps(_valid_response([extra]), ensure_ascii=False)
        with (
            patch.object(director, "_request_ollama", return_value=response) as request,
            patch.object(director, "_unload_after_run") as unload,
        ):
            tags, lyrics, report, plan_json = director.OnekoAceSongVariationDirector().direct(**_kwargs())

        self.assertEqual(request.call_count, 1)
        self.assertEqual(request.call_args.kwargs["num_predict"], 2200)
        self.assertFalse(request.call_args.kwargs["reasoning"])
        unload.assert_called_once_with(director.DEFAULT_OLLAMA_URL, director.DEFAULT_DIRECTOR_MODEL, 180, True)
        self.assertTrue(tags.startswith(SOURCE_TAGS))
        self.assertIn("Arrangement variation:", tags)
        self.assertNotIn("In Notmark steht der Baron schwer.", lyrics)
        self.assertIn("[Strophe 1 | VARIATION:", lyrics)
        self.assertIn("[Instrumental Bridge | VARIATION:", lyrics)
        self.assertLess(lyrics.index("[Refrain | VARIATION:"), lyrics.index("[Instrumental Bridge | VARIATION:"))
        self.assertLess(lyrics.index("[Instrumental Bridge | VARIATION:"), lyrics.index("[Verse 2 | VARIATION:"))
        for term in ("Notmark", "Baron", "Warzensau"):
            self.assertIn(term, lyrics)
        self.assertIn("3 source section(s), 1 added", report)
        plan = json.loads(plan_json)
        self.assertEqual(plan["status"], "ok")
        self.assertEqual([item["id"] for item in plan["arranged_sections"]], ["S01", "S02", "A01", "S03"])
        self.assertEqual(plan["arranged_sections"][2]["origin"], "additional")

    def test_invalid_initial_response_gets_exactly_one_repair(self):
        invalid = json.dumps({"global_arrangement": "missing the other keys"})
        repaired = json.dumps(_valid_response(), ensure_ascii=False)
        with (
            patch.object(director, "_request_ollama", side_effect=[invalid, repaired]) as request,
            patch.object(director, "_unload_after_run"),
        ):
            result = director.OnekoAceSongVariationDirector().direct(**_kwargs())

        self.assertEqual(request.call_count, 2)
        self.assertEqual(json.loads(result[3])["status"], "ok")
        repair_call = request.call_args_list[1]
        self.assertEqual(repair_call.args[4], 0.0)
        self.assertEqual(repair_call.args[5], 1.0)

    def test_missing_exact_term_twice_uses_passthrough(self):
        invalid = _valid_response()
        invalid["source_sections"][0]["lyrics"] = "Zu notmark zählt der Herr sein Gold."
        invalid["source_sections"][1]["lyrics"] = "Hebt an und ruft, er wird bald fallen!"
        raw = json.dumps(invalid, ensure_ascii=False)
        with (
            patch.object(director, "_request_ollama", side_effect=[raw, raw]) as request,
            patch.object(director, "_unload_after_run"),
        ):
            tags, lyrics, report, plan_json = director.OnekoAceSongVariationDirector().direct(**_kwargs())

        self.assertEqual(request.call_count, 2)
        self.assertEqual(tags, SOURCE_TAGS)
        self.assertEqual(lyrics, SOURCE_LYRICS)
        self.assertIn("passed through", report)
        self.assertEqual(json.loads(plan_json)["status"], "passthrough")

    def test_transport_failure_uses_passthrough_without_repair(self):
        with (
            patch.object(director, "_request_ollama", side_effect=RuntimeError("offline")) as request,
            patch.object(director, "_unload_after_run") as unload,
        ):
            result = director.OnekoAceSongVariationDirector().direct(**_kwargs())

        self.assertEqual(request.call_count, 1)
        self.assertEqual(result[:2], (SOURCE_TAGS, SOURCE_LYRICS))
        self.assertEqual(json.loads(result[3])["status"], "passthrough")
        unload.assert_called_once()

    def test_strict_mode_raises_after_one_failed_repair(self):
        invalid = "not JSON"
        with (
            patch.object(director, "_request_ollama", side_effect=[invalid, invalid]) as request,
            patch.object(director, "_unload_after_run"),
        ):
            with self.assertRaisesRegex(RuntimeError, "ACE Song Variation Director failed"):
                director.OnekoAceSongVariationDirector().direct(**_kwargs(fallback_mode="strict"))
        self.assertEqual(request.call_count, 2)

    def test_validator_rejects_extra_sections_and_hidden_headers(self):
        sections = director._parse_sections(SOURCE_LYRICS)
        result = _valid_response(
            [
                {"insert_after": "S01", "header": "Bridge", "direction": "quiet", "lyrics": "A"},
                {"insert_after": "S02", "header": "Bridge 2", "direction": "loud", "lyrics": "B"},
            ]
        )
        with self.assertRaisesRegex(ValueError, "at most 1"):
            director._validate_result(result, sections, ["Notmark", "Baron", "Warzensau"], 1)

        result = _valid_response()
        result["source_sections"][0]["lyrics"] += "\n[Secret Chorus]"
        with self.assertRaisesRegex(ValueError, "undeclared section header"):
            director._validate_result(result, sections, ["Notmark", "Baron", "Warzensau"], 1)

    def test_validator_expands_short_directions_and_requires_silent_instrumentals(self):
        sections = director._parse_sections(SOURCE_LYRICS)
        result = _valid_response()
        result["source_sections"][0]["direction"] = "Verse 1"
        validated = director._validate_result(result, sections, ["Notmark", "Baron", "Warzensau"], 1)
        self.assertIn("distinct contrasting arrangement", validated["source_sections"][0]["direction"])
        self.assertIn("Strophe 1", validated["source_sections"][0]["direction"])

        result = _valid_response(
            [
                {
                    "insert_after": "S02",
                    "header": "Instrumental Bridge",
                    "direction": "brief guitar break with rising drum accents",
                    "lyrics": "This production note must not be sung.",
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "must not contain sung lyrics"):
            director._validate_result(result, sections, ["Notmark", "Baron", "Warzensau"], 1)

    def test_validator_enforces_requested_lyrics_language(self):
        sections = director._parse_sections(SOURCE_LYRICS)
        result = _valid_response()
        for item in result["source_sections"]:
            item["lyrics"] = "The shadows rise while the cold wind crosses the road."
        with self.assertRaisesRegex(ValueError, "not German"):
            director._validate_result(result, sections, [], 1, "de")

    def test_is_changed_covers_text_and_variation_controls(self):
        base = _kwargs()
        digest = director.OnekoAceSongVariationDirector.IS_CHANGED(**base)
        for field, value in {
            "tags": "different tags",
            "lyrics": "different lyrics",
            "must_keep": "Notmark",
            "seed": 99,
            "energy_variation": 0.1,
            "max_new_sections": 0,
            "lyrics_language": "en",
        }.items():
            changed = dict(base)
            changed[field] = value
            self.assertNotEqual(
                digest,
                director.OnekoAceSongVariationDirector.IS_CHANGED(**changed),
                field,
            )


if __name__ == "__main__":
    unittest.main()
