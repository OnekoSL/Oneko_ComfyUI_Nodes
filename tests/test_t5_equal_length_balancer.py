import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


from tests.paths import COMFY_ROOT, ONEKO_ROOT
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

from Oneko_ComfyUI_Nodes.nodes.t5_equal_length_balancer import (
    OnekoT5EqualLengthBalancer,
    SUPPORTED_TEXT_STREAM_KEYS,
)
from Oneko_ComfyUI_Nodes.nodes.t5_sculpt_equal_length_balancer import (
    OnekoT5SculptEqualLengthBalancer,
)
from Oneko_ComfyUI_Nodes.nodes import embedding_sculpt_core as sculpt_core


class FakeKreaClip:
    def __init__(self, cond_stage_model=None):
        self.cond_stage_model = cond_stage_model or SimpleNamespace(qwen3vl_4b=object())
        self.options = {}

    def clone(self):
        return FakeKreaClip(self.cond_stage_model)

    def set_tokenizer_option(self, name, value):
        self.options[name] = value

    def tokenize(self, text):
        raw_length = len(text.split()) + 8
        target = self.options.get("qwen3vl_4b_min_length", 0)
        length = max(raw_length, target)
        return {"qwen3vl_4b": [[(token_id, 1.0) for token_id in range(length)]]}

    def encode_from_tokens_scheduled(self, tokens):
        return tokens


class FakeSculptKreaClip:
    def __init__(self, weight=None):
        self.weight = weight if weight is not None else torch.randn(64, 8, generator=torch.Generator().manual_seed(7))
        embedding = SimpleNamespace(weight=self.weight)
        transformer = SimpleNamespace(get_input_embeddings=lambda: embedding)
        self.cond_stage_model = SimpleNamespace(
            qwen3vl_4b=SimpleNamespace(
                transformer=transformer,
                special_tokens={"start": 0, "end": 63, "pad": 1},
            )
        )
        self.options = {}

    def clone(self):
        clone = FakeSculptKreaClip(self.weight)
        clone.options = dict(self.options)
        return clone

    def set_tokenizer_option(self, name, value):
        self.options[name] = value

    def tokenize(self, text):
        word_ids = [2 + (sum(word.encode("utf-8")) % 50) for word in text.split()]
        batch = [
            (151644, 1.0),
            (872, 1.0),
            (198, 1.0),
            *[(token_id, 1.0) for token_id in word_ids],
            (151645, 1.0),
        ]
        target = int(self.options.get("qwen3vl_4b_min_length", 0))
        batch.extend([(1, 1.0)] * max(0, target - len(batch)))
        return {"qwen3vl_4b": [batch]}

    def encode_from_tokens_scheduled(self, tokens):
        return tokens


class Qwen3VL4BBalancerTests(unittest.TestCase):
    def test_qwen3vl_4b_is_a_supported_stream(self):
        self.assertIn("qwen3vl_4b", SUPPORTED_TEXT_STREAM_KEYS)

    def test_equal_length_balancer_detects_and_pads_krea2_stream(self):
        result = OnekoT5EqualLengthBalancer().balance(
            FakeKreaClip(),
            16,
            "one two three",
            "one",
        )

        positive, negative, positive_raw, negative_raw, effective, report = result
        self.assertEqual((positive_raw, negative_raw, effective), (11, 9, 16))
        self.assertEqual(len(positive["qwen3vl_4b"][0]), 16)
        self.assertEqual(len(negative["qwen3vl_4b"][0]), 16)
        self.assertIn("text stream: qwen3vl_4b", report)

    def test_sculpt_balancer_selects_krea2_embedding_model(self):
        transformer = SimpleNamespace(get_input_embeddings=lambda: None)
        clip = SimpleNamespace(
            cond_stage_model=SimpleNamespace(
                qwen3vl_4b=SimpleNamespace(transformer=transformer)
            )
        )
        tokens = {"qwen3vl_4b": [[(1, 1.0)]]}

        stream = OnekoT5SculptEqualLengthBalancer()._detect_sculptable_text_stream_key(
            clip, tokens
        )

        self.assertEqual(stream, "qwen3vl_4b")

    def test_sculpting_excludes_krea2_chat_template(self):
        batch = [
            (151644, 1.0),
            (8948, 1.0),
            (198, 1.0),
            (1000, 1.0),
            (151645, 1.0),
            (198, 1.0),
            (151644, 1.0),
            (872, 1.0),
            (198, 1.0),
            (2000, 1.0),
            (2001, 1.0),
            (151645, 1.0),
            (198, 1.0),
            (151644, 1.0),
            (77091, 1.0),
            (198, 1.0),
            (151643, 1.0),
        ]
        tokens = {"qwen3vl_4b": [batch]}

        coords = OnekoT5SculptEqualLengthBalancer()._eligible_entries(
            tokens, "qwen3vl_4b", {151643}
        )

        self.assertEqual(coords, [(0, 9, 2000), (0, 10, 2001)])

    def test_positive_and_negative_share_one_low_memory_search(self):
        node = OnekoT5SculptEqualLengthBalancer()
        with patch.object(
            sculpt_core,
            "chunked_top_neighbors",
            wraps=sculpt_core.chunked_top_neighbors,
        ) as search:
            result = node.balance(
                FakeSculptKreaClip(),
                16,
                "red fox crystal",
                "blurred hands",
                0.35,
                "forward",
                "none",
                0.25,
                "backward",
                "mean",
                4,
            )

        self.assertEqual(search.call_count, 1)
        self.assertEqual(result[4], 16)
        self.assertIn("search device:", result[-1])
        self.assertIn("unique tokens:", result[-1])

    def test_target_default_remains_1024(self):
        target = OnekoT5SculptEqualLengthBalancer.INPUT_TYPES()["required"]["target"]
        self.assertEqual(target[1]["default"], 1024)


if __name__ == "__main__":
    unittest.main()
