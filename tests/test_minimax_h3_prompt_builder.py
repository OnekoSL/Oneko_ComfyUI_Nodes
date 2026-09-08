import sys
import unittest
from pathlib import Path
from unittest.mock import patch


from tests.paths import COMFY_ROOT, ONEKO_ROOT
CUSTOM_NODE_ROOT = ONEKO_ROOT
RESOURCE_ROOT = CUSTOM_NODE_ROOT / "resources"
for import_path in (COMFY_ROOT, CUSTOM_NODE_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from Oneko_ComfyUI_Nodes.nodes import minimax_h3_prompt_builder as h3


def _base_kwargs():
    kwargs = {
        "spoken_dialogue": "",
        "dialogue_language": "German",
        "dialogue_voice": "clear young female voice",
        "dialogue_delivery": "natural speech, calm and slightly cautious tone",
    }
    for section_name, _section_label in h3.SECTIONS:
        kwargs[f"{section_name}_text"] = ""
        kwargs[f"{section_name}_vocab_file"] = "resources/missing.txt"
        kwargs[f"{section_name}_amount"] = 0
        kwargs[f"{section_name}_word_index"] = 0
    return kwargs


class MiniMaxH3PromptBuilderTests(unittest.TestCase):
    def test_node_registration_inputs_and_outputs(self):
        self.assertIs(
            h3.NODE_CLASS_MAPPINGS["OnekoMiniMaxH3PromptBuilder"],
            h3.OnekoMiniMaxH3PromptBuilder,
        )
        self.assertEqual(
            h3.NODE_DISPLAY_NAME_MAPPINGS["OnekoMiniMaxH3PromptBuilder"],
            "MiniMax H3 Prompt Builder (Oneko)",
        )
        self.assertEqual(h3.OnekoMiniMaxH3PromptBuilder.CATEGORY, "Oneko/02 Text/Vocabulary")
        self.assertEqual(
            h3.OnekoMiniMaxH3PromptBuilder.RETURN_NAMES,
            ("prompt", "scene", "character", "action", "camera", "visual_style", "audio"),
        )

        inputs = h3.OnekoMiniMaxH3PromptBuilder.INPUT_TYPES()["required"]
        for section_name, _section_label in h3.SECTIONS:
            self.assertIn(f"{section_name}_text", inputs)
            self.assertEqual(inputs[f"{section_name}_amount"][1]["default"], 0)
            self.assertTrue(inputs[f"{section_name}_word_index"][1]["control_after_generate"])
            self.assertEqual(
                inputs[f"{section_name}_vocab_file"][1]["default"],
                h3.DEFAULT_SECTION_VOCAB_FILES[section_name],
            )

    def test_headers_have_fixed_order_and_empty_sections_are_omitted(self):
        kwargs = _base_kwargs()
        kwargs.update(
            {
                "scene_text": "A moonlit forest.",
                "action_text": "Sparkles drift through the air.",
                "audio_text": "A soft breeze.",
            }
        )

        result = h3.OnekoMiniMaxH3PromptBuilder().generate(**kwargs)

        self.assertEqual(
            result[0],
            "[Scene]\nA moonlit forest.\n\n"
            "[Action]\nSparkles drift through the air.\n\n"
            "[Audio]\nA soft breeze.",
        )
        self.assertNotIn("[Character]", result[0])
        self.assertNotIn("[Camera]", result[0])
        self.assertNotIn("[Visual Style]", result[0])

    def test_fixed_text_and_multiword_vocab_phrases_are_combined_cleanly(self):
        kwargs = _base_kwargs()
        kwargs.update(
            {
                "scene_text": "A forest at twilight",
                "scene_amount": 2,
                "scene_word_index": 0,
            }
        )
        words = ["glowing plants", "faint magical mist", "drifting sparkles"]
        expected_phrases = h3._shuffle_bag_sample(words, 2, 0, 1)

        with patch.object(h3, "_load_words", return_value=words):
            result = h3.OnekoMiniMaxH3PromptBuilder().generate(**kwargs)

        expected_scene = f"A forest at twilight, {', '.join(expected_phrases)}"
        self.assertEqual(result[1], expected_scene)
        self.assertEqual(result[0], f"[Scene]\n{expected_scene}")

    def test_amount_zero_does_not_load_vocab_and_keeps_fixed_text(self):
        kwargs = _base_kwargs()
        kwargs["character_text"] = "A young anime mage."

        with patch.object(h3, "_load_words", side_effect=AssertionError("vocab should not load")):
            result = h3.OnekoMiniMaxH3PromptBuilder().generate(**kwargs)

        self.assertEqual(result[2], "A young anime mage.")

    def test_sections_use_independent_shuffle_orders(self):
        words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
        selections = [tuple(h3._shuffle_bag_sample(words, 3, 0, index)) for index in range(1, 7)]

        self.assertGreater(len(set(selections)), 1)
        for selection in selections:
            self.assertEqual(len(selection), 3)
            self.assertEqual(len(set(selection)), 3)

    def test_dialogue_is_added_verbatim_to_action_and_audio(self):
        kwargs = _base_kwargs()
        kwargs.update(
            {
                "action_text": "She turns toward the camera",
                "audio_text": "Quiet magical forest ambience.",
                "spoken_dialogue": "Hallo, ist da jemand?",
            }
        )

        result = h3.OnekoMiniMaxH3PromptBuilder().generate(**kwargs)

        expected_line = '"Hallo, ist da jemand?"'
        self.assertEqual(result[3].count(expected_line), 1)
        self.assertEqual(result[6].count(expected_line), 1)
        self.assertIn(f"The character says in German: {expected_line}", result[3])
        self.assertIn(f"A clear young female voice says in German: {expected_line}", result[6])
        self.assertIn("Natural speech, calm and slightly cautious tone.", result[6])
        self.assertTrue(result[6].endswith("No other dialogue."))

    def test_dialogue_creates_action_and_audio_sections_on_its_own(self):
        kwargs = _base_kwargs()
        kwargs["spoken_dialogue"] = "Hallo!"

        result = h3.OnekoMiniMaxH3PromptBuilder().generate(**kwargs)

        self.assertEqual(result[0].count("[Action]"), 1)
        self.assertEqual(result[0].count("[Audio]"), 1)
        self.assertNotIn("[Scene]", result[0])

    def test_prequoted_multi_speaker_dialogue_is_not_wrapped_again(self):
        kwargs = _base_kwargs()
        dialogue = 'Frau: "nein, bitte nicht!" danach Mann: "sei still!!"'
        kwargs["spoken_dialogue"] = dialogue

        result = h3.OnekoMiniMaxH3PromptBuilder().generate(**kwargs)

        self.assertIn(dialogue, result[3])
        self.assertIn(dialogue, result[6])
        self.assertNotIn('"Frau: "', result[3])
        self.assertEqual(result[3].count('"'), 4)
        self.assertEqual(result[6].count('"'), 4)

    def test_is_changed_covers_sections_dialogue_and_cursors(self):
        base = _base_kwargs()
        base_digest = h3.OnekoMiniMaxH3PromptBuilder.IS_CHANGED(**base)

        changes = {
            "scene_text": "new scene",
            "camera_word_index": 8,
            "visual_style_amount": 2,
            "spoken_dialogue": "Hallo!",
            "dialogue_language": "English",
            "dialogue_voice": "deep narrator voice",
            "dialogue_delivery": "whispered",
        }
        for field_name, value in changes.items():
            changed = dict(base)
            changed[field_name] = value
            self.assertNotEqual(
                base_digest,
                h3.OnekoMiniMaxH3PromptBuilder.IS_CHANGED(**changed),
                field_name,
            )

    def test_complete_anime_mage_example(self):
        kwargs = _base_kwargs()
        kwargs.update(
            {
                "scene_text": (
                    "A young anime mage stands in a glowing forest at twilight. The forest is filled "
                    "with soft luminous plants, drifting sparkles, and faint magical mist. The atmosphere "
                    "feels mysterious, calm, and slightly enchanted."
                ),
                "character_text": (
                    "She has long blue hair, a white and blue fantasy outfit, and a flowing cape. She holds "
                    "a small glowing magical orb in one hand. Her expression is calm and confident, with a "
                    "hint of curiosity."
                ),
                "action_text": (
                    "Sparkling particles drift through the air as her hair and cape move gently in the wind. "
                    "She slowly raises her hand slightly, looks around the glowing forest, then turns toward "
                    "the camera."
                ),
                "camera_text": (
                    "Medium shot, subtle slow camera movement, slightly cinematic framing, gentle forward drift."
                ),
                "visual_style_text": (
                    "Beautiful anime style, clean line art, vibrant but soft fantasy colors, detailed magical "
                    "forest background, glowing effects, soft cinematic twilight lighting, smooth motion."
                ),
                "audio_text": (
                    "Quiet magical forest ambience with a soft breeze, faint rustling leaves, subtle sparkling "
                    "magical sounds, and distant nighttime forest atmosphere. Gentle fantasy anime soundtrack "
                    "with soft piano, airy pads, and light strings."
                ),
                "spoken_dialogue": "Hallo, ist da jemand?",
            }
        )

        result = h3.OnekoMiniMaxH3PromptBuilder().generate(**kwargs)

        expected_headers = ["[Scene]", "[Character]", "[Action]", "[Camera]", "[Visual Style]", "[Audio]"]
        positions = [result[0].index(header) for header in expected_headers]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(result[0].count('"Hallo, ist da jemand?"'), 2)
        self.assertEqual(result[1], kwargs["scene_text"])
        self.assertEqual(result[2], kwargs["character_text"])
        self.assertTrue(result[6].endswith("No other dialogue."))


class MiniMaxH3ResourceTests(unittest.TestCase):
    def test_each_section_has_eighty_unique_prompt_ready_phrases(self):
        selectable = set(h3._available_vocab_files())
        for section_name, resource_label in h3.DEFAULT_SECTION_VOCAB_FILES.items():
            self.assertIn(resource_label, selectable, section_name)
            entries = h3._load_words(resource_label)
            self.assertEqual(len(entries), 80, section_name)
            self.assertEqual(len({entry.casefold() for entry in entries}), 80, section_name)
            for entry in entries:
                self.assertEqual(entry, entry.strip())
                self.assertNotIn("\n", entry)
                self.assertNotIn("  ", entry)
                self.assertGreaterEqual(len(entry.split()), 8)
                self.assertLessEqual(len(entry.split()), 25)

    def test_actions_are_continuous_character_movements(self):
        entries = h3._load_words(h3.DEFAULT_SECTION_VOCAB_FILES["action"])
        for entry in entries:
            self.assertTrue(entry.startswith("The character "), entry)
            self.assertGreaterEqual(len(entry.split()), 8, entry)

    def test_camera_phrases_avoid_cuts_and_montages(self):
        entries = h3._load_words(h3.DEFAULT_SECTION_VOCAB_FILES["camera"])
        forbidden = ("jump cut", "rapid cut", "montage", "scene change", "teleport")
        for entry in entries:
            lowered = entry.casefold()
            self.assertFalse(any(marker in lowered for marker in forbidden), entry)

    def test_audio_phrases_do_not_invent_spoken_dialogue(self):
        entries = h3._load_words(h3.DEFAULT_SECTION_VOCAB_FILES["audio"])
        forbidden = ('"', " says ", "spoken dialogue", "narrator voice", "voiceover")
        for entry in entries:
            lowered = entry.casefold()
            self.assertFalse(any(marker in lowered for marker in forbidden), entry)

    def test_visual_styles_avoid_named_artist_language(self):
        entries = h3._load_words(h3.DEFAULT_SECTION_VOCAB_FILES["visual_style"])
        forbidden = ("inspired by", "in the style of", " franchise")
        for entry in entries:
            lowered = entry.casefold()
            self.assertFalse(any(marker in lowered for marker in forbidden), entry)


if __name__ == "__main__":
    unittest.main()
