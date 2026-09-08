import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


from tests.paths import COMFY_ROOT, ONEKO_ROOT
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

from Oneko_ComfyUI_Nodes.nodes import clip_sculpt_core as clip_core
from Oneko_ComfyUI_Nodes.nodes import embedding_sculpt_core as core
from Oneko_ComfyUI_Nodes.nodes.regional_sculpt_prompt_encoder import (
    OnekoRegionalSculptPromptEncoder,
)


def dense_neighbors(weight, token_ids, top_k):
    normalized = torch.nn.functional.normalize(weight.float(), dim=1)
    result = {}
    for token_id in token_ids:
        scores = torch.matmul(normalized, normalized[token_id])
        scores[token_id] = -torch.inf
        values, ids = torch.topk(scores, k=min(top_k, weight.shape[0] - 1))
        result[token_id] = (ids.tolist(), values.tolist())
    return result


def dense_refine(weight, token_id, method, intensity, top_k):
    neighbors = dense_neighbors(weight, [token_id], top_k)[token_id]
    return core.refine_from_neighbors(
        weight,
        token_id,
        neighbors[0],
        neighbors[1],
        method,
        intensity,
    )


class FakeClip:
    def __init__(self, weight):
        embedding = SimpleNamespace(weight=weight)
        transformer = SimpleNamespace(get_input_embeddings=lambda: embedding)
        submodel = SimpleNamespace(
            transformer=transformer,
            special_tokens={"start": 0, "end": weight.shape[0] - 1},
        )
        self.cond_stage_model = SimpleNamespace(l=submodel)

    def tokenize(self, text):
        token_ids = [2 + (sum(word.encode("utf-8")) % (self.cond_stage_model.l.transformer.get_input_embeddings().weight.shape[0] - 3)) for word in text.split()]
        end = self.cond_stage_model.l.transformer.get_input_embeddings().weight.shape[0] - 1
        return {"l": [[(0, 1.0), *[(token_id, 1.0) for token_id in token_ids], (end, 1.0)]]}

    def encode_from_tokens_scheduled(self, tokens):
        return tokens


class FakeDualClip:
    def __init__(self, weight):
        def submodel():
            embedding = SimpleNamespace(weight=weight)
            return SimpleNamespace(
                transformer=SimpleNamespace(get_input_embeddings=lambda: embedding),
                special_tokens={"start": 0, "end": weight.shape[0] - 1},
            )

        self.cond_stage_model = SimpleNamespace(l=submodel(), g=submodel())


class ChunkedEmbeddingSearchTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(1234)
        self.weight = torch.randn(31, 9, generator=generator)

    def test_chunked_top_k_matches_dense_across_chunk_boundaries(self):
        token_ids = [1, 7, 16, 29]
        expected = dense_neighbors(self.weight, token_ids, 6)
        actual, stats = core._search_on_device(
            self.weight,
            token_ids,
            6,
            torch.device("cpu"),
            work_bytes=180,
            query_batch_size=2,
        )
        self.assertGreater(stats["query_batches"], 1)
        self.assertLess(stats["chunk_rows"], self.weight.shape[0])
        for token_id in token_ids:
            self.assertEqual(actual[token_id][0], expected[token_id][0])
            self.assertTrue(
                torch.allclose(
                    torch.tensor(actual[token_id][1]),
                    torch.tensor(expected[token_id][1]),
                    atol=1e-6,
                )
            )
            self.assertNotIn(token_id, actual[token_id][0])

    def test_all_refinement_methods_match_dense_reference(self):
        for method in ("forward", "backward", "maximum_absolute", "add_minimum_absolute"):
            expected, expected_count = dense_refine(self.weight, 7, method, 0.35, 8)
            session = core.EmbeddingSculptSession(
                self.weight,
                [7],
                8,
                preferred_device=torch.device("cpu"),
            )
            actual, actual_count = session.sculpt(7, method, 0.35)
            self.assertEqual(actual_count, expected_count)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6), method)

    def test_accelerator_oom_retries_exact_search_on_cpu(self):
        real_search = core._search_on_device
        calls = []

        def fake_search(weight, token_ids, top_k, device, **kwargs):
            device = torch.device(device)
            calls.append(device.type)
            if device.type != "cpu":
                raise torch.cuda.OutOfMemoryError("forced test OOM")
            return real_search(weight, token_ids, top_k, device, **kwargs)

        with (
            patch.object(core, "_search_on_device", side_effect=fake_search),
            patch.object(core.model_management, "soft_empty_cache") as empty_cache,
        ):
            actual, stats = core.chunked_top_neighbors(
                self.weight,
                [3, 9],
                5,
                preferred_device=torch.device("cuda"),
            )

        self.assertEqual(calls, ["cuda", "cpu"])
        self.assertTrue(stats["cpu_fallback"])
        self.assertEqual(actual[3][0], dense_neighbors(self.weight, [3], 5)[3][0])
        empty_cache.assert_called_once()

    def test_non_oom_errors_are_not_hidden(self):
        with patch.object(core, "_search_on_device", side_effect=ValueError("bad embedding")):
            with self.assertRaisesRegex(ValueError, "bad embedding"):
                core.chunked_top_neighbors(
                    self.weight,
                    [2],
                    3,
                    preferred_device=torch.device("cuda"),
                )

    def test_streaming_mean_magnitude_matches_dense_mean(self):
        expected = torch.linalg.vector_norm(self.weight.float(), dim=1).mean().item()
        actual, stats = core.streaming_mean_magnitude(
            self.weight,
            preferred_device=torch.device("cpu"),
        )
        self.assertAlmostEqual(actual, expected, places=6)
        self.assertEqual(stats["device"], "cpu")

    def test_multiple_clip_token_sets_share_one_search_session(self):
        clip = FakeClip(self.weight)
        token_sets = [
            {"l": [[(0, 1.0), (2, 1.0), (4, 1.0), (30, 1.0)]]},
            {"l": [[(0, 1.0), (4, 1.0), (6, 1.0), (30, 1.0)]]},
        ]
        with patch.object(core, "chunked_top_neighbors", wraps=core.chunked_top_neighbors) as search:
            sculpted, stats = clip_core.sculpt_clip_token_sets(
                clip,
                token_sets,
                0.5,
                "forward",
                "mean",
                4,
                strict=True,
            )
        self.assertEqual(search.call_count, 1)
        self.assertEqual(stats[0]["eligible"], 2)
        self.assertEqual(stats[1]["eligible"], 2)
        self.assertTrue(torch.is_tensor(sculpted[0]["l"][0][1][0]))
        self.assertIn("unique tokens: 3", stats[0]["search"])

    def test_dual_clip_streams_each_use_one_bounded_search(self):
        clip = FakeDualClip(self.weight)
        tokens = {
            "l": [[(0, 1.0), (2, 1.0), (3, 1.0), (30, 1.0)]],
            "g": [[(0, 1.0), (4, 1.0), (5, 1.0), (30, 1.0)]],
        }
        with patch.object(core, "chunked_top_neighbors", wraps=core.chunked_top_neighbors) as search:
            _sets, stats = clip_core.sculpt_clip_token_sets(
                clip, [tokens], 0.5, "forward", "none", 4, strict=True
            )
        self.assertEqual(search.call_count, 2)
        self.assertEqual(stats[0]["eligible"], 4)
        self.assertIn("l: 2 eligible", stats[0]["streams"])
        self.assertIn("g: 2 eligible", stats[0]["streams"])

    def test_regional_encoder_reuses_one_search_across_all_texts(self):
        clip = FakeClip(self.weight)
        with patch.object(core, "chunked_top_neighbors", wraps=core.chunked_top_neighbors) as search:
            result = OnekoRegionalSculptPromptEncoder().encode(
                clip,
                "cinematic scene",
                "red fox",
                "blue crystal",
                "",
                2,
                ", ",
                "sharp detail",
                0.5,
                "forward",
                "mean",
                4,
            )
        self.assertEqual(search.call_count, 1)
        self.assertEqual(result[8], "")
        self.assertIn("search device:", result[-1])

    def test_regional_node_interface_is_unchanged(self):
        self.assertEqual(len(OnekoRegionalSculptPromptEncoder.RETURN_TYPES), 11)
        self.assertEqual(len(OnekoRegionalSculptPromptEncoder.RETURN_NAMES), 11)
        required = OnekoRegionalSculptPromptEncoder.INPUT_TYPES()["required"]
        self.assertEqual(required["top_k"][1]["default"], 64)


if __name__ == "__main__":
    unittest.main()
