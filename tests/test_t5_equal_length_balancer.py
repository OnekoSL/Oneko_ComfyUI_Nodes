import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
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


class FakeQwenChatTokenizer:
    """Chat metadata for the small-vocabulary Qwen3VL sculpt test double."""

    _markers = {"<|im_start|>": 151644, "<|im_end|>": 151645,
                "<|endoftext|>": 151643, "<think>": 151667, "</think>": 151668}

    all_special_ids = [151643, 151644, 151645]

    def decode(self, ids):
        return "".join("\n" if token == 198 else "unknown" for token in ids)

    def get_added_vocab(self):
        return dict(self._markers)

    def encode(self, text, add_special_tokens=False):
        pieces = {**self._markers, "system": 8948, "user": 872, "assistant": 77091, "\n": 198}
        result = []
        while text:
            for piece, token_id in pieces.items():
                if text.startswith(piece):
                    result.append(token_id)
                    text = text[len(piece):]
                    break
            else:
                raise ValueError(f"Unsupported fake metadata text: {text!r}")
        return result


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
        self.tokenizer = SimpleNamespace(
            qwen3vl_4b=SimpleNamespace(tokenizer=FakeQwenChatTokenizer())
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
            tokens, "qwen3vl_4b", {151643}, clip=FakeSculptKreaClip()
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


class TokenIdEmbedding(torch.nn.Module):
    """Tiny deterministic embedding stand-in; native Core still builds the mask."""

    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.zeros(1, 1))

    def forward(self, tokens, out_dtype=torch.float32):
        return tokens.to(out_dtype).unsqueeze(-1)


class IdentityTextTransformer(torch.nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        self.num_layers = 1
        self.embedding = TokenIdEmbedding()

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, tokens, attention_mask, *, embeds, **kwargs):
        return embeds, None


class NativeTokenizerClip:
    """Real local tokenizer and Core mask/encoding path, without model weights."""

    def __init__(self, tokenizer, stream="qwen35_9b", special_tokens=None):
        from comfy.sd1_clip import SDClipModel

        self.tokenizer = tokenizer
        self.stream = stream
        self.options = {}
        self.encoded = []
        self.cond_stage_model = SimpleNamespace(**{
            stream: SDClipModel(
                textmodel_json_config={},
                model_class=IdentityTextTransformer,
                special_tokens=special_tokens or {"pad": 248044},
                enable_attention_masks=True,
                return_attention_masks=True,
            )
        })

    def clone(self):
        clone = object.__new__(type(self))
        clone.__dict__ = self.__dict__.copy()
        clone.options = dict(self.options)
        return clone

    def set_tokenizer_option(self, name, value):
        self.options[name] = value

    def tokenize(self, text):
        return self.tokenizer.tokenize_with_weights(text, tokenizer_options=self.options)

    def encode_from_tokens_scheduled(self, tokens):
        model = getattr(self.cond_stage_model, self.stream)
        cond, pooled, extra = model.encode_token_weights(tokens[self.stream])
        result = [[cond, {"pooled_output": pooled, **extra}]]
        self.encoded.append((tokens, result))
        return result


@pytest.fixture(scope="module")
def moss_tokenizer():
    from comfy.text_encoders.moss_image import MOSSImageTokenizer

    return MOSSImageTokenizer()


@pytest.mark.parametrize("positive,negative,target", [
    ("A red fox beside a moss-covered ruin.", "", 0),
    ("", "", 0),
    ("", "", 4096),
    ("a red fox", "blurred", 1),
    ("a red fox", "blurred", 4096),
    ("", "A red fox beside a moss-covered ruin.", 0),
    ("blurred", "a red fox", 4096),
    ("A fox rests by the ancient tree. " * 600, "bad anatomy", 16),
], ids=["empty-negative", "both-empty", "empty-large-target", "below-raw", "above-raw",
        "empty-positive", "negative-longer", "long-prompt"])
def test_moss_preserves_native_lengths_and_conditioning(moss_tokenizer, positive, negative, target):
    clip = NativeTokenizerClip(moss_tokenizer)
    # Existing caller options must not affect raw measurement or be overwritten.
    clip.options = {"qwen35_9b_min_length": 128, "qwen35_9b_min_padding": 7}
    original_options = dict(clip.options)
    original_min_length = moss_tokenizer.qwen35_9b.min_length
    original_min_padding = moss_tokenizer.qwen35_9b.min_padding
    raw = [moss_tokenizer.tokenize_with_weights(text, tokenizer_options={
        "qwen35_9b_min_length": 0, "qwen35_9b_min_padding": 0,
    })["qwen35_9b"][0] for text in (positive, negative)]

    pos, neg, pos_raw, neg_raw, effective, report = OnekoT5EqualLengthBalancer().balance(
        clip, target, positive, negative,
    )

    assert (pos_raw, neg_raw) == tuple(map(len, raw))
    assert effective == max(map(len, raw))
    assert "MOSS native lengths" in report
    assert "requested target not applied" in report
    assert "text stream: qwen35_9b" in report
    assert "including chat template" in report
    assert "excluding extra padding" in report
    for index, (native_tokens, conditioning) in enumerate(zip(raw, (pos, neg))):
        encoded_tokens = clip.encoded[index][0]["qwen35_9b"][0]
        assert encoded_tokens == native_tokens
        assert len(encoded_tokens) == len(native_tokens)
        native_conditioning = clip.encoded[index][1]
        assert conditioning[0][0] is native_conditioning[0][0]
        context = conditioning[0][1]["oneko_conditioning_context"]
        assert context["version"] == 1
        assert context["model_profile"] == "moss_image"
        assert context["token_count"] == len(native_tokens)
        assert len(context["token_mask"]) == len(native_tokens)
        assert all(type(flag) is bool for flag in context["token_mask"])
        assert set(conditioning[0][1]) == set(native_conditioning[0][1]) | {"oneko_conditioning_context"}
        tensor, metadata = conditioning[0]
        assert tensor.shape == (1, len(native_tokens), 1)
        expected_mask = torch.ones(1, len(native_tokens), dtype=torch.long)
        torch.testing.assert_close(metadata["attention_mask"], expected_mask)
        # All native tokens, including role/think markers, survive encoding in order.
        assert tensor[0, :, 0].tolist() == [t[0] for t in native_tokens]
    assert clip.options == original_options
    assert moss_tokenizer.qwen35_9b.min_length == original_min_length
    assert moss_tokenizer.qwen35_9b.min_padding == original_min_padding


def test_moss_raw_count_includes_complete_chat_template(moss_tokenizer):
    empty = moss_tokenizer.tokenize_with_weights("")["qwen35_9b"][0]
    ids = [entry[0] for entry in empty]
    text = moss_tokenizer.qwen35_9b.tokenizer.decode(ids, skip_special_tokens=False)
    assert text == "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<think>\n"
    assert 248044 not in ids
    result = OnekoT5EqualLengthBalancer().balance(NativeTokenizerClip(moss_tokenizer), 0, "", "")
    assert result[2:5] == (len(ids), len(ids), len(ids))


@pytest.mark.parametrize("node_type", [OnekoT5EqualLengthBalancer, OnekoT5SculptEqualLengthBalancer])
def test_generic_qwen35_keeps_native_padding(node_type):
    from comfy.text_encoders.qwen35 import Qwen35ImageTokenizer

    # The stream name alone does not identify MOSS's diffusion contract.
    tokenizer = Qwen35ImageTokenizer(model_type="qwen35_9b")
    clip = NativeTokenizerClip(tokenizer)
    args = (clip, 64, "red fox", "")
    if node_type is OnekoT5SculptEqualLengthBalancer:
        args += (0.0, "forward", "none", 0.0, "backward", "none", 4)
    result = node_type().balance(*args)
    assert result[4] == 64
    assert "MOSS native lengths" not in result[-1]
    for raw_length, (tokens, conditioning) in zip(result[2:4], clip.encoded):
        batch = tokens["qwen35_9b"][0]
        assert len(batch) == 64
        assert batch[raw_length:] == [(248044, 1.0)] * (64 - raw_length)
        assert conditioning[0][0].shape == (1, 64, 1)
        assert conditioning[0][1]["attention_mask"].tolist() == [[1] * raw_length + [0] * (64 - raw_length)]


@pytest.mark.parametrize("sculpt_enabled", [False, True], ids=["plain", "sculpt"])
def test_legacy_balancer_works_without_moss_core_module(sculpt_enabled):
    if sculpt_enabled:
        node = OnekoT5SculptEqualLengthBalancer()
        args = (FakeSculptKreaClip(), 16, "red fox", "blurred", 0.0, "forward", "none", 0.0, "backward", "none", 4)
    else:
        node = OnekoT5EqualLengthBalancer()
        args = (FakeKreaClip(), 16, "red fox", "blurred")
    with patch.dict(sys.modules, {"comfy.text_encoders.moss_image": None}):
        result = node.balance(*args)
    assert result[4] == 16
    assert all(len(conditioning["qwen3vl_4b"][0]) == 16 for conditioning in result[:2])
    assert "MOSS native lengths" not in result[-1]


def test_moss_core_keeps_different_context_lengths_separate(moss_tokenizer):
    from comfy.conds import CONDRegular
    from comfy.model_base import BaseModel, MOSSImage

    clip = NativeTokenizerClip(moss_tokenizer)
    result = OnekoT5EqualLengthBalancer().balance(clip, 4096, "red fox in an ancient forest", "")
    # Execute real Core condition handling without constructing model weights.
    model = MOSSImage.__new__(MOSSImage)
    with patch.object(MOSSImage, "concat_cond", return_value=None), patch.object(BaseModel, "encode_adm", return_value=None):
        model_conditions = [model.extra_conds(cross_attn=cond[0][0], **cond[0][1]) for cond in result[:2]]
    positive, negative = [conditions["c_crossattn"] for conditions in model_conditions]
    assert type(positive) is type(negative) is CONDRegular
    assert positive.cond is result[0][0][0]
    assert negative.cond is result[1][0][0]
    assert not positive.can_concat(negative)
    # Encoder attention masks do not mask the MOSS diffusion's context tokens.
    assert all("attention_mask" not in conditions for conditions in model_conditions)


def test_t5_end_token_padding_and_other_streams_are_preserved():
    from comfy.sd1_clip import SD1Tokenizer
    from comfy.text_encoders.sd3_clip import T5XXLTokenizer

    tokenizer = SD1Tokenizer(name="t5xxl", tokenizer=T5XXLTokenizer)

    class MixedTokenizer:
        def tokenize_with_weights(self, text, **kwargs):
            return {"l": [[(49406, 1.0), (42, 1.0), (49407, 1.0)]],
                    **tokenizer.tokenize_with_weights(text, **kwargs)}

    clip = NativeTokenizerClip(MixedTokenizer(), "t5xxl", {"end": 1, "pad": 0})
    pos, neg, pos_raw, neg_raw, effective, _ = OnekoT5EqualLengthBalancer().balance(
        clip, 32, "red fox", "",
    )
    assert effective == 32
    assert neg_raw == 1  # T5's native EOS token, also for an empty prompt.
    for (tokens, _), raw_count, conditioning in zip(clip.encoded, (pos_raw, neg_raw), (pos, neg)):
        t5 = tokens["t5xxl"][0]
        assert t5[raw_count - 1] == (1, 1.0)
        assert t5[raw_count:] == [(0, 1.0)] * (effective - raw_count)
        assert tokens["l"] == [[(49406, 1.0), (42, 1.0), (49407, 1.0)]]
        assert conditioning[0][1]["attention_mask"].tolist() == [[1] * raw_count + [0] * (effective - raw_count)]


def test_conditioning_stream_uses_present_submodel_with_mixed_tokens():
    clip = SimpleNamespace(cond_stage_model=SimpleNamespace(qwen35_9b=object()))
    tokens = {"t5xxl": [[(1, 1.0)]], "qwen35_9b": [[(248045, 1.0)]]}
    assert OnekoT5EqualLengthBalancer()._detect_conditioning_text_stream_key(clip, tokens) == "qwen35_9b"


if __name__ == "__main__":
    unittest.main()
