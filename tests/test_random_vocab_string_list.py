import importlib.util
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


from tests.paths import COMFY_ROOT, ONEKO_ROOT
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

MODULE_PATH = ONEKO_ROOT / "nodes" / "random_vocab_string_list.py"
RESOURCE_ROOT = MODULE_PATH.parents[1] / "resources"
spec = importlib.util.spec_from_file_location("random_vocab_string_list", MODULE_PATH)
vocab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vocab)


class MultiVocabShuffleBagTests(unittest.TestCase):
    def test_random_node_uses_shuffle_bag_blocks(self):
        node = vocab.OnekoRandomVocabStringList()
        with patch.object(vocab, "_load_words", return_value=["alpha", "beta", "gamma", "delta"]):
            first = node.generate("resources/test.txt", 2, 0)[0]
            second = node.generate("resources/test.txt", 2, 1)[0]

        words = first.split() + second.split()
        self.assertEqual(len(words), 4)
        self.assertEqual(set(words), {"alpha", "beta", "gamma", "delta"})

    def test_random_node_dedupes_vocab_before_sampling(self):
        node = vocab.OnekoRandomVocabStringList()
        with patch.object(vocab, "_load_words", return_value=["alpha", "beta", "alpha", "gamma", "beta", "delta"]):
            first = node.generate("resources/test.txt", 2, 0)[0]
            second = node.generate("resources/test.txt", 2, 1)[0]

        words = first.split() + second.split()
        self.assertEqual(len(words), 4)
        self.assertEqual(len(words), len(set(words)))
        self.assertEqual(set(words), {"alpha", "beta", "gamma", "delta"})

    def test_random_node_is_changed_includes_chain(self):
        first = vocab.OnekoRandomVocabStringList.IS_CHANGED("resources/missing.txt", 2, 0, chain="alpha")
        second = vocab.OnekoRandomVocabStringList.IS_CHANGED("resources/missing.txt", 2, 0, chain="beta")

        self.assertNotEqual(first, second)

    def test_word_index_amount_one_covers_bag_before_repeating(self):
        words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
        picks = [vocab._shuffle_bag_sample(words, 1, word_index, 1)[0] for word_index in range(len(words))]

        self.assertEqual(len(picks), len(words))
        self.assertEqual(set(picks), set(words))

    def test_amount_greater_than_one_uses_non_overlapping_blocks(self):
        words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
        picks = []
        for word_index in range(3):
            picks.extend(vocab._shuffle_bag_sample(words, 2, word_index, 1))

        self.assertEqual(len(picks), len(words))
        self.assertEqual(set(picks), set(words))

    def test_cycle_wrap_uses_next_shuffled_bag(self):
        words = ["alpha", "beta", "gamma", "delta", "epsilon"]
        first_cycle = list(words)
        second_cycle = list(words)
        random.Random(vocab._slot_seed(0, 1)).shuffle(first_cycle)
        random.Random(vocab._slot_seed(1, 1)).shuffle(second_cycle)

        self.assertEqual(vocab._shuffle_bag_sample(words, 3, 0, 1), first_cycle[:3])
        self.assertEqual(
            vocab._shuffle_bag_sample(words, 3, 1, 1),
            first_cycle[3:] + second_cycle[:1],
        )
        self.assertNotEqual(first_cycle, second_cycle)

    def test_generate_slot_uses_word_index_as_block_cursor(self):
        with patch.object(vocab, "_load_words", return_value=["alpha", "beta", "gamma", "delta"]):
            first = vocab._generate_slot("resources/test.txt", 2, 0, 1)
            second = vocab._generate_slot("resources/test.txt", 2, 1, 1)

        words = first.split() + second.split()
        self.assertEqual(len(words), 4)
        self.assertEqual(set(words), {"alpha", "beta", "gamma", "delta"})

    def test_amount_zero_disables_slot(self):
        with patch.object(vocab, "_load_words", return_value=["alpha", "beta"]):
            self.assertEqual(vocab._generate_slot("resources/test.txt", 0, 0, 1), "")

    def test_duplicate_vocab_entries_do_not_increase_selection_weight(self):
        words = ["alpha", "beta", "alpha", "gamma", "beta", "delta"]
        deduped = vocab._dedupe_preserve_order(words)
        picks = []
        for seed in range(2):
            picks.extend(vocab._shuffle_bag_sample(words, 2, seed, 1))

        self.assertEqual(deduped, ["alpha", "beta", "gamma", "delta"])
        self.assertEqual(len(picks), len(set(picks)))
        self.assertEqual(set(picks), set(deduped))

    def test_slots_use_different_shuffle_orders(self):
        words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
        self.assertNotEqual(
            vocab._shuffle_bag_sample(words, 3, 0, 1),
            vocab._shuffle_bag_sample(words, 3, 0, 2),
        )

    def test_multi_node_inputs_use_word_indices_not_word_selectors(self):
        inputs = vocab.OnekoVocabMultiStringList.INPUT_TYPES()["required"]

        self.assertNotIn("seed", inputs)
        for slot in range(1, vocab.MULTI_SLOT_COUNT + 1):
            self.assertIn(f"word_index_{slot}", inputs)
            self.assertEqual(inputs[f"word_index_{slot}"][1]["control_after_generate"], True)
            self.assertEqual(
                inputs[f"word_index_{slot}"][1]["control_prefix"],
                f"word_index_{slot}",
            )
            for word_index in range(1, 4):
                self.assertNotIn(f"word_{slot}_{word_index}", inputs)

        self.assertEqual(
            len(
                {
                    inputs[f"word_index_{slot}"][1]["control_prefix"]
                    for slot in range(1, vocab.MULTI_SLOT_COUNT + 1)
                }
            ),
            vocab.MULTI_SLOT_COUNT,
        )


class VisualArtStylesResourceTests(unittest.TestCase):
    RESOURCE_LABEL = "resources/visual_art_styles.csv"
    REPRESENTATIVE_STYLES = {
        "continuous line drawing",
        "gouache illustration",
        "linocut print style",
        "graphic novel art",
        "anime style",
        "typographic poster",
        "pixel art",
        "stained glass art",
        "film noir cinematography",
        "art nouveau illustration",
    }

    @classmethod
    def _resource_text(cls):
        return (RESOURCE_ROOT / "visual_art_styles.csv").read_text(encoding="utf-8")

    def test_visual_art_styles_is_selectable(self):
        self.assertIn(self.RESOURCE_LABEL, vocab._list_resource_vocab_files())

    def test_visual_art_styles_has_ten_groups_of_twenty_five(self):
        lines = self._resource_text().splitlines()
        self.assertEqual(len(lines), 10)
        for line in lines:
            self.assertEqual(len([entry for entry in line.split(",") if entry]), 25)

    def test_visual_art_styles_are_unique_prompt_ready_phrases(self):
        entries = vocab._load_words(self.RESOURCE_LABEL)
        self.assertEqual(len(entries), 250)
        self.assertEqual(len({entry.casefold() for entry in entries}), 250)

        for entry in entries:
            self.assertEqual(entry, entry.strip())
            self.assertEqual(entry, entry.lower())
            self.assertNotIn("  ", entry)
            self.assertNotIn("\n", entry)
            self.assertLessEqual(len(entry.split()), 6)

    def test_visual_art_styles_cover_all_planned_categories(self):
        entries = set(vocab._load_words(self.RESOURCE_LABEL))
        self.assertTrue(self.REPRESENTATIVE_STYLES.issubset(entries))

    def test_visual_art_styles_avoid_named_style_language(self):
        entries = vocab._load_words(self.RESOURCE_LABEL)
        forbidden_markers = ("inspired by", "in the style of", "studio ", " franchise")
        for entry in entries:
            lowered = entry.casefold()
            self.assertFalse(any(marker in lowered for marker in forbidden_markers))


if __name__ == "__main__":
    unittest.main()
