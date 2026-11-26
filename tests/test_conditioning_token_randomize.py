import pytest
import torch

from Oneko_ComfyUI_Nodes.nodes.clip_sculpt_text_encode import OnekoCLIPSculptTextEncode
from Oneko_ComfyUI_Nodes.nodes.conditioning_context import (
    CONDITIONING_CONTEXT_KEY,
    conditioning_fingerprint,
)
from Oneko_ComfyUI_Nodes.nodes.conditioning_token_randomize import OnekoConditioningTokenRandomize
from tests.test_t5_equal_length_balancer import IdentityTextTransformer, NativeTokenizerClip, moss_tokenizer


def test_zero_strength_returns_original_conditioning():
    conditioning = [[torch.ones(1, 4, 6), {"pooled_output": torch.ones(1, 6)}]]

    output = OnekoConditioningTokenRandomize.execute(conditioning, 0.0, 42, 0, 0)

    assert output[0] is conditioning


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_full_strength_replaces_only_selected_tokens_at_their_original_norm(dtype):
    tensor = torch.arange(1, 49, dtype=dtype).reshape(2, 4, 6)
    tensor[0, 1] = 0
    original = tensor.clone()
    metadata = {
        "pooled_output": torch.ones(2, 6),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "nested": {"example": True},
        "start_percent": 0.2,
    }

    output = OnekoConditioningTokenRandomize.execute([[tensor, metadata]], 1.0, 42, 1, 2)
    result, result_metadata = output[0][0]

    assert (result.shape, result.dtype, result.device) == (tensor.shape, tensor.dtype, tensor.device)
    torch.testing.assert_close(result[:, (0, 3)], tensor[:, (0, 3)], rtol=0, atol=0)
    assert not torch.equal(result[:, 1:3], tensor[:, 1:3])
    torch.testing.assert_close(result[0, 1], torch.zeros_like(result[0, 1]), rtol=0, atol=0)
    torch.testing.assert_close(
        result[:, 1:3].float().norm(dim=-1), tensor[:, 1:3].float().norm(dim=-1),
        rtol=0.005, atol=0,
    )
    torch.testing.assert_close(tensor, original, rtol=0, atol=0)
    assert result_metadata is not metadata
    assert result_metadata.keys() == metadata.keys()
    for key in metadata:
        assert result_metadata[key] is metadata[key]


def test_seed_is_local_and_repeatable_across_matching_entries():
    tensor = torch.arange(1, 25, dtype=torch.float32).reshape(1, 4, 6)
    conditioning = [[tensor, {"start_percent": 0.2}], [tensor.clone(), {"start_percent": 0.6}]]
    rng_state = torch.random.get_rng_state()
    seed = 0xFFFFFFFF

    result = OnekoConditioningTokenRandomize.execute(conditioning, 1.0, seed, 0, 0)[0]
    same_seed = OnekoConditioningTokenRandomize.execute(conditioning, 1.0, seed, 0, 0)[0]
    other_seed = OnekoConditioningTokenRandomize.execute(conditioning, 1.0, 3, 0, 0)[0]

    torch.testing.assert_close(result[0][0], result[1][0], rtol=0, atol=0)
    torch.testing.assert_close(result[0][0], same_seed[0][0], rtol=0, atol=0)
    assert not torch.equal(result[0][0], other_seed[0][0])
    assert torch.equal(torch.random.get_rng_state(), rng_state)


def test_interpolation_and_token_range_through_end():
    tensor = torch.arange(1, 25, dtype=torch.float32).reshape(1, 4, 6)
    conditioning = [[tensor, {}]]

    full = OnekoConditioningTokenRandomize.execute(conditioning, 1.0, 8, 2, 0)[0][0][0]
    oversized = OnekoConditioningTokenRandomize.execute(conditioning, 1.0, 8, 2, 100)[0][0][0]
    partial = OnekoConditioningTokenRandomize.execute(conditioning, 0.25, 8, 2, 0)[0][0][0]

    torch.testing.assert_close(full, oversized, rtol=0, atol=0)
    torch.testing.assert_close(partial[:, :2], tensor[:, :2], rtol=0, atol=0)
    torch.testing.assert_close(partial[:, 2:], torch.lerp(tensor[:, 2:], full[:, 2:], 0.25))


def test_metadata_only_unsupported_and_out_of_range_entries_pass_through():
    metadata = {"pooled_output": torch.ones(1, 4)}
    conditioning = [
        [None, metadata],
        [torch.ones(1, 3, 4, dtype=torch.long), metadata],
        [torch.ones(1, 4), metadata],
        [torch.ones(1, 1, 4), metadata],
    ]

    result = OnekoConditioningTokenRandomize.execute(conditioning, 1.0, 14, 1, 0)[0]

    assert len(result) == len(conditioning)
    for updated, original in zip(result, conditioning):
        assert updated[0] is original[0]
        assert updated[1] is original[1]


def test_native_anima_encoded_conditioning_keeps_t5_ids_and_weights(monkeypatch):
    from comfy.text_encoders import anima, llama

    class SmallQwen(IdentityTextTransformer):
        def forward(self, tokens, attention_mask, *, embeds, **kwargs):
            return embeds.expand(-1, -1, 1024), None

    monkeypatch.setattr(llama, "Qwen3_06B", SmallQwen)
    tokens = anima.AnimaTokenizer().tokenize_with_weights("a red (fox:1.4) under a blue umbrella")
    tensor, pooled, extra = anima.AnimaTEModel().encode_token_weights(tokens)
    metadata = {"pooled_output": pooled, **extra}

    result, result_metadata = OnekoConditioningTokenRandomize.execute([[tensor, metadata]], 0.5, 42, 1, 2)[0][0]

    assert tensor.shape == result.shape == (1, len(tokens["qwen3_06b"][0]), 1024)
    assert result_metadata["t5xxl_ids"].tolist() == [token_id for token_id, _ in tokens["t5xxl"][0]]
    torch.testing.assert_close(
        result_metadata["t5xxl_weights"], torch.tensor([weight for _, weight in tokens["t5xxl"][0]]),
    )
    assert not torch.equal(result[:, 1:3], tensor[:, 1:3])
    torch.testing.assert_close(result[:, :1], tensor[:, :1], rtol=0, atol=0)
    torch.testing.assert_close(result[:, 3:], tensor[:, 3:], rtol=0, atol=0)
    for key in metadata:
        assert result_metadata[key] is metadata[key]


@pytest.fixture
def moss_conditioning(moss_tokenizer):
    clip = NativeTokenizerClip(moss_tokenizer)
    conditioning, _ = OnekoCLIPSculptTextEncode().encode(
        clip, "A red fox sits in heavy rain.", 0.0, "forward", "none", 4,
    )
    return conditioning


def test_verified_moss_edits_only_user_text_in_the_absolute_range_and_rebinds(moss_conditioning):
    tensor, metadata = moss_conditioning[0]
    metadata["reference_latents"] = [torch.ones(1, 128, 2, 2)]
    metadata["context_attention_bias"] = torch.zeros(1, tensor.shape[1])
    context = metadata[CONDITIONING_CONTEXT_KEY]
    user_positions = [index for index, editable in enumerate(context["token_mask"]) if editable]
    selected_position = user_positions[1]
    original_fingerprint = context["source_fingerprint"]

    output = OnekoConditioningTokenRandomize.execute(moss_conditioning, 0.5, 4, selected_position, 1)[0]
    edited, edited_metadata = output[0]

    changed_positions = (edited != tensor).any(dim=(0, 2)).nonzero().flatten().tolist()
    assert changed_positions == [selected_position]
    assert edited_metadata[CONDITIONING_CONTEXT_KEY] == {
        **context, "source_fingerprint": conditioning_fingerprint(edited),
    }
    assert conditioning_fingerprint(edited) != original_fingerprint
    assert context["source_fingerprint"] == conditioning_fingerprint(tensor) == original_fingerprint
    for key in metadata.keys() - {CONDITIONING_CONTEXT_KEY}:
        assert edited_metadata[key] is metadata[key]


def test_verified_moss_leaves_metadata_untouched_when_range_has_no_editable_positions(moss_conditioning):
    entry = moss_conditioning[0]
    assert entry[1][CONDITIONING_CONTEXT_KEY]["token_mask"][0] is False

    output = OnekoConditioningTokenRandomize.execute(moss_conditioning, 1.0, 4, 0, 1)[0]

    assert output[0] is entry


def test_verified_moss_does_not_rebind_when_selected_values_are_unchanged(moss_conditioning):
    tensor, metadata = moss_conditioning[0]
    tensor = torch.zeros_like(tensor)
    context = {**metadata[CONDITIONING_CONTEXT_KEY], "source_fingerprint": conditioning_fingerprint(tensor)}
    entry = [tensor, {**metadata, CONDITIONING_CONTEXT_KEY: context}]

    output = OnekoConditioningTokenRandomize.execute([entry], 1.0, 4, 0, 0)[0]

    assert output[0] is entry


@pytest.mark.parametrize("reason", ["stale", "unavailable"])
def test_moss_invalid_provenance_cannot_fall_back_to_unprotected_edit(moss_conditioning, caplog, reason):
    tensor, metadata = moss_conditioning[0]
    if reason == "stale":
        tensor = tensor + 1
    else:
        context = {**metadata[CONDITIONING_CONTEXT_KEY], "token_mask": None}
        metadata = {**metadata, CONDITIONING_CONTEXT_KEY: context}
    entry = [tensor, metadata]

    output = OnekoConditioningTokenRandomize.execute([entry], 1.0, 4, 0, 0)[0]

    assert output[0] is entry
    assert "skipped" in caplog.text and "MOSS" in caplog.text


def test_native_moss_without_position_metadata_preserves_generic_positional_editing(moss_tokenizer):
    clip = NativeTokenizerClip(moss_tokenizer)
    native = clip.encode_from_tokens_scheduled(clip.tokenize("A red fox sits in heavy rain."))
    tensor, metadata = native[0]

    edited, edited_metadata = OnekoConditioningTokenRandomize.execute(native, 0.5, 4, 0, 1)[0][0]

    assert (edited != tensor).any(dim=(0, 2)).nonzero().flatten().tolist() == [0]
    assert CONDITIONING_CONTEXT_KEY not in edited_metadata
    assert edited_metadata["attention_mask"] is metadata["attention_mask"]
