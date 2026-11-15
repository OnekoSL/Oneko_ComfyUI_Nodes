from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from Oneko_ComfyUI_Nodes.nodes import clip_sculpt_core as sculpt
from Oneko_ComfyUI_Nodes.nodes import embedding_sculpt_core as search_core
from Oneko_ComfyUI_Nodes.nodes.clip_sculpt_text_encode import OnekoCLIPSculptTextEncode
from Oneko_ComfyUI_Nodes.nodes.regional_sculpt_prompt_encoder import OnekoRegionalSculptPromptEncoder
from Oneko_ComfyUI_Nodes.nodes.t5_equal_length_balancer import OnekoT5EqualLengthBalancer
from Oneko_ComfyUI_Nodes.nodes.t5_sculpt_equal_length_balancer import OnekoT5SculptEqualLengthBalancer
from tests.test_t5_equal_length_balancer import NativeTokenizerClip, moss_tokenizer


@pytest.fixture(scope="module", params=["moss", "qwen3vl"])
def chat_tokenizer(request, moss_tokenizer):
    if request.param == "moss":
        return moss_tokenizer, "qwen35_9b", 248044
    from comfy.text_encoders.qwen3vl import Qwen3VLTokenizer
    return Qwen3VLTokenizer(model_type="qwen3vl_4b"), "qwen3vl_4b", 151643


def chat_coords(tokenizer, stream, pad, text):
    clip = SimpleNamespace(tokenizer=tokenizer)
    tokens = tokenizer.tokenize_with_weights(text)
    coords = sculpt.eligible_entries(tokens, stream, {pad}, clip=clip)
    return tokens, coords


@pytest.mark.parametrize("text,expected", [
    ("red fox", "red fox"),
    ("", ""),
    ("<|im_start|>user\n\nred fox<|im_end|>\n<|im_start|>assistant\n<think>\n", "red fox"),
    ("<|im_start|>system\nred fox<|im_end|>\n"
     "<|im_start|>user\nred fox<|im_end|>\n"
     "<|im_start|>assistant\nred fox<|im_end|>\n"
     "<|im_start|>user\nblue bird<|im_end|>\n"
     "<|im_start|>assistant\n<think>red fox", "red foxblue bird"),
    ("<think>red</think><|vision_start|><|image_pad|><|vision_end|>", "red"),
], ids=["plain", "empty", "merged-newlines", "multiple-roles", "special-inside-user"])
def test_only_user_text_is_eligible(chat_tokenizer, text, expected):
    tokenizer, stream, pad = chat_tokenizer
    tokens, coords = chat_coords(tokenizer, stream, pad, text)
    hf = getattr(tokenizer, stream).tokenizer
    assert hf.decode([token for _, _, token in coords]) == expected
    # Passing formatted input through native tokenization must not wrap it again.
    if text.startswith("<|im_start|>"):
        assert [entry[0] for entry in tokens[stream][0]] == hf.encode(text, add_special_tokens=False)
    assert not (set(token for _, _, token in coords) & set(hf.get_added_vocab().values()))


@pytest.mark.parametrize("text", [
    "<|im_start|>user\nunfinished",
    "<|im_start|>user\nred fox<|im_start|>assistant\ntext<|im_end|>",
    "<|im_start|>system\nunfinished<|im_start|>user\nred fox<|im_end|>",
    "<|im_start|>unknown\nred fox<|im_end|>",
    "<|im_start|>user red fox<|im_end|>",
], ids=["missing-end", "nested-turn", "unclosed-system", "unknown-role", "missing-separator"])
def test_unrecognized_or_incomplete_chat_is_not_sculpted(chat_tokenizer, text):
    tokenizer, stream, pad = chat_tokenizer
    assert chat_coords(tokenizer, stream, pad, text)[1] == []


def test_unstructured_chat_and_missing_tokenizer_are_not_sculpted(chat_tokenizer):
    tokenizer, stream, pad = chat_tokenizer
    raw = getattr(tokenizer, stream).tokenizer.encode("red fox", add_special_tokens=False)
    tokens = {stream: [[(token, 1.0) for token in raw]]}
    assert sculpt.eligible_entries(tokens, stream, {pad}, SimpleNamespace(tokenizer=tokenizer)) == []
    assert sculpt.eligible_entries(tokens, stream, {pad}) == []


def test_multimodal_entries_remain_untouched(chat_tokenizer):
    tokenizer, stream, pad = chat_tokenizer
    hf = getattr(tokenizer, stream).tokenizer
    image = {"type": "image", "data": torch.zeros(1, 1, 1, 3)}
    vector = torch.ones(8)
    prefix = hf.encode("<|im_start|>user\n", add_special_tokens=False)
    suffix = hf.encode("red fox<|im_end|>", add_special_tokens=False)
    batch = [(token, 1.0) for token in prefix] + [(image, 1.0), (vector, 1.0)] + [(token, 1.0) for token in suffix]
    coords = sculpt.eligible_entries({stream: [batch]}, stream, {pad}, SimpleNamespace(tokenizer=tokenizer))
    assert hf.decode([token for _, _, token in coords]) == "red fox"
    assert batch[len(prefix)][0] is image
    assert batch[len(prefix) + 1][0] is vector


class TableEmbedding(torch.nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.register_buffer("weight", torch.randn(vocab, 8, generator=torch.Generator().manual_seed(71)))

    def forward(self, tokens, out_dtype=torch.float32):
        return torch.nn.functional.embedding(tokens, self.weight).to(out_dtype)


@pytest.fixture(scope="module")
def moss_embedding(moss_tokenizer):
    return TableEmbedding(len(moss_tokenizer.qwen35_9b.tokenizer))


def moss_clip(tokenizer, embedding):
    clip = NativeTokenizerClip(tokenizer)
    clip.cond_stage_model.qwen35_9b.transformer.embedding = embedding
    return clip


def assert_conditioning_equal(actual, expected):
    torch.testing.assert_close(actual[0][0], expected[0][0], rtol=0, atol=0)
    # Oneko may add provenance, while all native metadata and numeric payloads
    # must remain exactly equal to direct Core encoding.
    key = "oneko_conditioning_context"
    for conditioning in (actual, expected):
        context = conditioning[0][1].get(key)
        if context is not None:
            assert context["version"] == 1
            assert context["model_profile"] == "moss_image"
            assert context["token_count"] == conditioning[0][0].shape[1]
            assert len(context["token_mask"]) == context["token_count"]
            assert all(type(flag) is bool for flag in context["token_mask"])
    actual_meta = {k: v for k, v in actual[0][1].items() if k != key}
    expected_meta = {k: v for k, v in expected[0][1].items() if k != key}
    assert actual_meta.keys() == expected_meta.keys()
    for key in actual_meta:
        if torch.is_tensor(actual_meta[key]):
            torch.testing.assert_close(actual_meta[key], expected_meta[key], rtol=0, atol=0)
        else:
            assert actual_meta[key] == expected_meta[key]


def run_path(path, clip, intensity, normalization):
    if path == "clip":
        return OnekoCLIPSculptTextEncode().encode(clip, "red fox", intensity, "forward", normalization, 4)
    if path == "balancer":
        return OnekoT5SculptEqualLengthBalancer().balance(
            clip, 32, "red fox", "", intensity, "forward", normalization,
            intensity, "backward", normalization, 4,
        )
    return OnekoRegionalSculptPromptEncoder().encode(
        clip, "red", "fox", "bird", "", 2, " ", "detail",
        intensity, "forward", normalization, 4,
    )


@pytest.mark.parametrize("path", ["clip", "balancer", "regional"])
def test_disabled_sculpt_is_identical_to_native_encoding(moss_tokenizer, moss_embedding, path):
    clip = moss_clip(moss_tokenizer, moss_embedding)
    result = run_path(path, clip, 0.0, "none")
    encoded = list(clip.encoded)
    for tokens, conditioning in encoded:
        assert all(isinstance(entry[0], int) for batch in tokens["qwen35_9b"] for entry in batch)
        assert_conditioning_equal(conditioning, clip.encode_from_tokens_scheduled(tokens))
    if path == "balancer":
        native = OnekoT5EqualLengthBalancer().balance(clip, 32, "red fox", "")
        for index, text in enumerate(("red fox", "")):
            expected = clip.encode_from_tokens_scheduled(clip.tokenize(text))
            assert_conditioning_equal(result[index], expected)
            assert_conditioning_equal(native[index], expected)
        assert result[2:5] == native[2:5]
    elif path == "clip":
        assert_conditioning_equal(result[0], clip.encode_from_tokens_scheduled(clip.tokenize("red fox")))
    else:
        for index, text in enumerate(result[5:10]):
            assert_conditioning_equal(result[index], clip.encode_from_tokens_scheduled(clip.tokenize(text or "red")))


@pytest.mark.parametrize("path", ["clip", "balancer", "regional"])
@pytest.mark.parametrize("intensity,normalization", [(0.35, "none"), (0.35, "mean"), (0.35, "mean * attention"), (0.0, "mean"), (0.0, "mean * attention")])
def test_sculpt_preserves_template_length_mask_and_determinism(moss_tokenizer, moss_embedding, path, intensity, normalization):
    clip = moss_clip(moss_tokenizer, moss_embedding)
    with patch.object(search_core, "chunked_top_neighbors", wraps=search_core.chunked_top_neighbors) as search:
        result = run_path(path, clip, intensity, normalization)
    assert search.call_count == 1
    first_encoded = list(clip.encoded)
    changed = 0
    for index, (tokens, conditioning) in enumerate(first_encoded):
        if path == "regional":
            texts = [result[6], result[7], result[9], result[5], result[5]]
        else:
            texts = ["red fox", ""] if path == "balancer" else ["red fox"]
        native = clip.tokenize(texts[index])
        native_batch = native["qwen35_9b"][0]
        batch = tokens["qwen35_9b"][0]
        coords = sculpt.eligible_entries(native, "qwen35_9b", {248044}, clip)
        expected_positions = {position for _, position, _ in coords}
        assert len(batch) == len(native_batch)
        for position, (actual, original) in enumerate(zip(batch, native_batch)):
            if torch.is_tensor(actual[0]):
                assert position in expected_positions
                changed += 1
                assert actual[1] == original[1]
            else:
                assert actual == original
        native_conditioning = clip.encode_from_tokens_scheduled(native)
        assert conditioning[0][0].shape == native_conditioning[0][0].shape
        torch.testing.assert_close(conditioning[0][1]["attention_mask"], native_conditioning[0][1]["attention_mask"])
        protected = [i for i in range(len(batch)) if i not in expected_positions]
        torch.testing.assert_close(conditioning[0][0][:, protected], native_conditioning[0][0][:, protected], rtol=0, atol=0)
    assert changed > 0
    clip.encoded.clear()
    run_path(path, clip, intensity, normalization)
    for (_, first), (_, second) in zip(first_encoded, clip.encoded):
        assert_conditioning_equal(first, second)


@pytest.mark.parametrize("target", [0, 4096], ids=["auto", "large-target"])
@pytest.mark.parametrize("intensity", [0.0, 0.35], ids=["neutral", "active"])
@pytest.mark.parametrize("positive,negative", [
    ("red fox beside the ancient tree", ""),
    ("red fox beside the ancient tree", "blurred"),
    ("blurred", "red fox beside the ancient tree"),
], ids=["empty-negative", "positive-longer", "negative-longer"])
def test_moss_sculpt_balancer_uses_native_lengths(moss_tokenizer, moss_embedding, target, intensity, positive, negative):
    clip = moss_clip(moss_tokenizer, moss_embedding)
    # Caller padding must not accidentally become the MOSS diffusion text length.
    clip.options = {"qwen35_9b_min_length": 128, "qwen35_9b_min_padding": 7}
    original_options = dict(clip.options)
    native_clip = moss_clip(moss_tokenizer, moss_embedding)
    native_tokens = [native_clip.tokenize(text) for text in (positive, negative)]
    native_counts = [len(tokens["qwen35_9b"][0]) for tokens in native_tokens]
    result = OnekoT5SculptEqualLengthBalancer().balance(
        clip, target, positive, negative,
        intensity, "forward", "none", intensity, "backward", "none", 4,
    )
    assert result[2:5] == (*native_counts, max(native_counts))
    assert "MOSS native lengths" in result[-1]
    assert "requested target not applied" in result[-1]
    assert clip.options == original_options
    assert len(clip.encoded) == 2
    changed = 0
    for index, (actual_tokens, actual_conditioning) in enumerate(clip.encoded):
        native = native_tokens[index]
        native_conditioning = native_clip.encode_from_tokens_scheduled(native)
        actual_batch = actual_tokens["qwen35_9b"][0]
        native_batch = native["qwen35_9b"][0]
        assert len(actual_batch) == len(native_batch)
        assert actual_conditioning[0][0].shape == native_conditioning[0][0].shape
        torch.testing.assert_close(actual_conditioning[0][1]["attention_mask"], native_conditioning[0][1]["attention_mask"])
        eligible = {position for _, position, _ in sculpt.eligible_entries(native, "qwen35_9b", {248044}, native_clip)}
        for position, (actual, original) in enumerate(zip(actual_batch, native_batch)):
            if intensity and position in eligible:
                assert torch.is_tensor(actual[0])
                assert actual[1] == original[1]
                changed += 1
            else:
                assert actual == original
        protected = [position for position in range(len(native_batch)) if position not in eligible]
        torch.testing.assert_close(actual_conditioning[0][0][:, protected], native_conditioning[0][0][:, protected], rtol=0, atol=0)
        if not intensity:
            assert_conditioning_equal(result[index], native_conditioning)
    assert (changed > 0) == bool(intensity)


def test_empty_moss_user_has_no_sculptable_tokens(moss_tokenizer, moss_embedding):
    clip = moss_clip(moss_tokenizer, moss_embedding)
    output, report = OnekoCLIPSculptTextEncode().encode(clip, "", 0.5, "forward", "mean", 4)
    assert "eligible: 0" in report
    assert_conditioning_equal(output, clip.encode_from_tokens_scheduled(clip.tokenize("")))


def test_t5_sculpt_still_excludes_eos_and_padding():
    from comfy.sd1_clip import SD1Tokenizer
    from comfy.text_encoders.sd3_clip import T5XXLTokenizer
    tokenizer = SD1Tokenizer(name="t5xxl", tokenizer=T5XXLTokenizer)
    clip = NativeTokenizerClip(tokenizer, "t5xxl", {"end": 1, "pad": 0})
    clip.cond_stage_model.t5xxl.transformer.embedding = TableEmbedding(32128)
    result = OnekoT5SculptEqualLengthBalancer().balance(
        clip, 16, "red fox", "", 0.35, "forward", "mean", 0, "forward", "none", 4,
    )
    positive = clip.encoded[0][0]["t5xxl"][0]
    assert all(torch.is_tensor(entry[0]) for entry in positive[:result[2] - 1])
    assert positive[result[2] - 1:] == [(1, 1.0)] + [(0, 1.0)] * (16 - result[2])
    assert result[4] == 16


def test_incomplete_chat_is_encoded_unchanged_with_report(moss_tokenizer, moss_embedding):
    clip = moss_clip(moss_tokenizer, moss_embedding)
    text = "<|im_start|>user\nunfinished"
    output, report = OnekoCLIPSculptTextEncode().encode(clip, text, 0.5, "forward", "mean", 4)
    assert "eligible: 0" in report
    assert "incomplete chat skipped" in report
    assert_conditioning_equal(output, clip.encode_from_tokens_scheduled(clip.tokenize(text)))
