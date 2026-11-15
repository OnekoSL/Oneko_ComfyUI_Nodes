from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from Oneko_ComfyUI_Nodes.nodes import conditioning_tools as adjust
from Oneko_ComfyUI_Nodes.nodes.conditioning_context import (
    CONDITIONING_CONTEXT_KEY,
    attach_conditioning_context,
    capture_conditioning_context,
    conditioning_fingerprint,
)
from tests.test_t5_equal_length_balancer import NativeTokenizerClip, moss_tokenizer


ACTIVE_PRESETS = ["literal_detail", "soft_balance", "contrast_pop", "negative_tamer"]
ALL_PRESETS = ["neutral_report_only", *ACTIVE_PRESETS]


def context(mask, tensor):
    return {
        "version": 1,
        "model_profile": "moss_image",
        "token_mask": tuple(mask),
        "token_count": len(mask),
        "warning": "",
        "source_fingerprint": conditioning_fingerprint(tensor),
    }


def metadata(mask, tensor):
    return {
        CONDITIONING_CONTEXT_KEY: context(mask, tensor),
        "attention_mask": torch.ones(1, len(mask), dtype=torch.long),
        "pooled_output": torch.tensor([[2.0, 3.0]]),
        "mask": torch.tensor([[0.0, 1.0], [0.5, 1.0]]),
        "reference_latents": [torch.ones(1, 128, 2, 2)],
        "start_percent": 0.2,
        "end_percent": 0.8,
        "nested": {"value": ["preserve"]},
    }


def equal_metadata(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        other = expected[key]
        if torch.is_tensor(value):
            torch.testing.assert_close(value, other, rtol=0, atol=0)
        elif isinstance(value, dict):
            equal_metadata(value, other)
        elif isinstance(value, list) and value and torch.is_tensor(value[0]):
            assert len(value) == len(other)
            for left, right in zip(value, other):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
        else:
            assert value == other


def assert_adjust_metadata(actual, original, result_tensor):
    native_actual = {k: v for k, v in actual.items() if k != CONDITIONING_CONTEXT_KEY}
    native_original = {k: v for k, v in original.items() if k != CONDITIONING_CONTEXT_KEY}
    equal_metadata(native_actual, native_original)
    current_context = actual[CONDITIONING_CONTEXT_KEY]
    old_context = original[CONDITIONING_CONTEXT_KEY]
    assert {k: v for k, v in current_context.items() if k != "source_fingerprint"} == {
        k: v for k, v in old_context.items() if k != "source_fingerprint"
    }
    assert current_context["source_fingerprint"] == conditioning_fingerprint(result_tensor)


def run(tensor, mask, preset="negative_tamer", strength=1.0,
        preserve=True, enabled=True, profile="auto"):
    return adjust.OnekoConditioningAdjust().adjust(
        [[tensor, metadata(mask, tensor)]], preset, strength, preserve, enabled, profile,
    )


def test_existing_schema_and_output_positions_are_preserved():
    required = adjust.OnekoConditioningAdjust.INPUT_TYPES()["required"]
    assert list(required) == [
        "conditioning", "preset", "strength", "preserve_magnitude", "enabled", "model_profile",
    ]
    assert list(required["preset"][0]) == ALL_PRESETS
    assert list(required["model_profile"][0])[:4] == ["auto", "t5", "sdxl_clip", "sd15_clip"]
    assert "moss_image" in required["model_profile"][0]
    assert required["model_profile"][1]["default"] == "auto"
    assert required["preset"][1]["default"] == "literal_detail"
    assert adjust.OnekoConditioningAdjust.RETURN_TYPES == ("CONDITIONING", "STRING")


@pytest.mark.parametrize("preset", ALL_PRESETS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("preserve", [False, True])
def test_moss_presets_preserve_protected_positions_metadata_dtype_and_rng(preset, dtype, preserve):
    mask = [False, True, True, False, True, True, True, False]
    tensor = torch.randn((2, 8, 64), generator=torch.Generator().manual_seed(2026)).to(dtype)
    tensor[:, 5] *= 6
    source = [[tensor, metadata(mask, tensor)]]
    original = deepcopy(source)
    rng_before = torch.random.get_rng_state()
    output, report = adjust.OnekoConditioningAdjust().adjust(
        source, preset, 1.0, preserve, True, "auto",
    )
    actual = output[0][0]
    assert actual.shape == tensor.shape
    assert actual.dtype == dtype
    assert actual.device == tensor.device
    assert torch.isfinite(actual).all()
    selected = torch.tensor(mask, dtype=torch.bool)
    torch.testing.assert_close(actual[:, ~selected], tensor[:, ~selected], rtol=0, atol=0)
    torch.testing.assert_close(source[0][0], original[0][0], rtol=0, atol=0)
    equal_metadata(source[0][1], original[0][1])
    assert_adjust_metadata(output[0][1], original[0][1], actual)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert "moss_image" in report
    if preset == "neutral_report_only":
        torch.testing.assert_close(actual, tensor, rtol=0, atol=0)
    else:
        assert not torch.equal(actual[:, selected], tensor[:, selected])
        assert "oneko_encoder" in report
        assert "editable" in report.lower()
        assert "protected" in report.lower()
    if preserve:
        before = tensor[:, selected].float().square().mean(dim=(1, 2)).sqrt()
        after = actual[:, selected].float().square().mean(dim=(1, 2)).sqrt()
        tolerance = {torch.float32: 1e-6, torch.float16: 8e-4, torch.bfloat16: 5e-3}[dtype]
        torch.testing.assert_close(after, before, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("preset", ACTIVE_PRESETS)
def test_moss_batch_members_do_not_affect_each_other(preset):
    mask = [False, True, True, False, True, True, False]
    first = torch.randn((1, 7, 16), generator=torch.Generator().manual_seed(19))
    second = torch.randn((1, 7, 16), generator=torch.Generator().manual_seed(20)) * 50
    separate = run(first, mask, preset)[0][0][0]
    combined = run(torch.cat([first, second]), mask, preset)[0][0][0]
    torch.testing.assert_close(combined[:1], separate, rtol=0, atol=0)


@pytest.mark.parametrize("preset", ACTIVE_PRESETS)
def test_moss_all_protected_or_zero_embeddings_are_finite_and_unchanged(preset):
    tensor = torch.randn((2, 9, 16), generator=torch.Generator().manual_seed(7))
    output, _ = run(tensor, [False] * 9, preset)
    torch.testing.assert_close(output[0][0], tensor, rtol=0, atol=0)
    zeros = torch.zeros_like(tensor)
    output, _ = run(zeros, [False, True, True, False, True, True, False, True, False], preset)
    torch.testing.assert_close(output[0][0], zeros, rtol=0, atol=0)
    assert torch.isfinite(output[0][0]).all()


@pytest.mark.parametrize("preset", ACTIVE_PRESETS)
def test_moss_protected_values_do_not_influence_editable_values(preset):
    mask = [False, True, True, False, True, True, True, False]
    selected = torch.tensor(mask)
    original = torch.randn((1, 8, 16), generator=torch.Generator().manual_seed(15))
    altered = original.clone()
    altered[:, ~selected] *= 1000
    first = run(original, mask, preset)[0][0][0]
    second = run(altered, mask, preset)[0][0][0]
    torch.testing.assert_close(first[:, selected], second[:, selected], rtol=0, atol=0)


def test_soft_balance_smoothing_never_crosses_protected_gaps(monkeypatch):
    monkeypatch.setitem(adjust.PRESET_PARAMS, "soft_balance", {
        "center": 0.0, "contrast": 0.0, "smooth": 0.35, "spike": 0.0,
    })
    mask = [False, True, True, False, True, True, False]
    direction = torch.tensor([1.0, -1.0, 1.0, -1.0])
    tensor = torch.tensor([3.0, 1.0, 8.0, 200.0, 2.0, 4.0, 3.0])[None, :, None] * direction
    changed_other_segment = tensor.clone()
    changed_other_segment[:, 4:6] *= 100
    first = run(tensor, mask, "soft_balance", preserve=False)[0][0][0]
    second = run(changed_other_segment, mask, "soft_balance", preserve=False)[0][0][0]
    torch.testing.assert_close(first[:, 1:3], second[:, 1:3], rtol=0, atol=0)
    assert not torch.equal(first[:, 1:3], tensor[:, 1:3])


@pytest.mark.parametrize("preset,spread_direction", [
    ("literal_detail", 1), ("contrast_pop", 1), ("soft_balance", -1), ("negative_tamer", -1),
])
def test_presets_retain_their_relative_token_norm_effect(preset, spread_direction):
    mask = [False, True, True, True, True, True, False]
    direction = torch.tensor([1.0, -1.0, 1.0, -1.0])
    tensor = torch.tensor([99.0, 1.0, 2.0, 3.0, 4.0, 30.0, 99.0])[None, :, None] * direction
    result = run(tensor, mask, preset)[0][0][0]
    norms_before = torch.linalg.vector_norm(tensor[:, 1:6], dim=-1)
    norms_after = torch.linalg.vector_norm(result[:, 1:6], dim=-1)
    spread_before = norms_before.std(unbiased=False) / norms_before.mean()
    spread_after = norms_after.std(unbiased=False) / norms_after.mean()
    assert (spread_after - spread_before) * spread_direction > 0


@pytest.mark.parametrize("mode", ["neutral", "disabled", "zero-strength"])
@pytest.mark.parametrize("profile", ["auto", "moss_image", "t5"])
def test_all_bypasses_are_exact_even_without_valid_provenance(mode, profile):
    tensor = torch.randn((1, 5, 4096), generator=torch.Generator().manual_seed(16))
    source = [[tensor, {CONDITIONING_CONTEXT_KEY: {"version": 999, "model_profile": "moss_image"}}]]
    preset = "neutral_report_only" if mode == "neutral" else "soft_balance"
    strength = 0.0 if mode == "zero-strength" else 1.0
    enabled = mode != "disabled"
    output, report = adjust.OnekoConditioningAdjust().adjust(
        source, preset, strength, True, enabled, profile,
    )
    torch.testing.assert_close(output[0][0], tensor, rtol=0, atol=0)
    equal_metadata(output[0][1], source[0][1])
    assert "bypass" in report.lower()


@pytest.mark.parametrize("bad_context", [
    None,
    {"version": 1, "model_profile": "moss_image", "token_count": 4, "token_mask": (False, True, True, False)},
    {"version": 2, "model_profile": "moss_image", "token_count": 4, "token_mask": (False, True, True, False)},
    {"version": 1, "model_profile": "moss_image", "token_count": 4, "token_mask": None},
    {"version": 1, "model_profile": "moss_image", "token_count": 4, "token_mask": (False, True)},
    {"version": 1, "model_profile": "moss_image", "token_count": 5, "token_mask": (False, True, True, False)},
    {"version": 1, "model_profile": "moss_image", "token_count": 4, "token_mask": (0, 1, 1, 0)},
], ids=["missing", "missing-fingerprint", "unknown-version", "missing-mask", "mask-length", "count-mismatch", "not-bool"])
def test_active_moss_requires_valid_provenance(bad_context):
    tensor = torch.ones(1, 4, 4096)
    meta = {} if bad_context is None else {CONDITIONING_CONTEXT_KEY: bad_context}
    with pytest.raises(ValueError):
        adjust.OnekoConditioningAdjust().adjust(
            [[tensor, meta]], "soft_balance", 1.0, True, True, "moss_image",
        )


@pytest.mark.parametrize("other_profile", ["t5", "sdxl_clip", "sd15_clip"])
def test_explicit_other_profile_cannot_override_known_moss_protection(other_profile):
    with pytest.raises(ValueError):
        run(torch.ones(1, 4, 4096), [False, True, True, False], profile=other_profile)


def test_auto_uses_encoder_provenance_and_does_not_claim_width_identifies_moss():
    tensor = torch.ones(1, 4, 4096)
    profile, source, _ = adjust._resolve_model_profile(tensor, metadata([False, True, True, False], tensor), "auto")
    assert (profile, source) == ("moss_image", "oneko_encoder")
    profile, source, warning = adjust._resolve_model_profile(tensor, {}, "auto")
    assert profile == "t5"  # Legacy heuristic is retained, but is not a MOSS detector.
    assert warning
    assert source != "oneko_encoder"


@pytest.mark.parametrize("text,expected_user_text", [
    ("red fox", "red fox"),
    ("", ""),
    ("<|im_start|>user\n\nred fox<|im_end|>\n<|im_start|>assistant\n<think>\n", "red fox"),
    ("<|im_start|>system\nkeep me<|im_end|>\n"
     "<|im_start|>user\nred fox<|im_end|>\n"
     "<|im_start|>assistant\nkeep me<|im_end|>\n"
     "<|im_start|>user\nblue bird<|im_end|>\n"
     "<|im_start|>assistant\n<think>\n", "red foxblue bird"),
    ("red <think>fox</think>", "red fox"),
    ("<|im_start|>user\nunfinished", ""),
], ids=["plain", "empty", "formatted", "multi-turn", "special-in-user", "malformed"])
def test_context_uses_actual_native_moss_token_positions(moss_tokenizer, text, expected_user_text):
    clip = NativeTokenizerClip(moss_tokenizer)
    tokens = clip.tokenize(text)
    captured = capture_conditioning_context(clip, tokens)
    assert captured["model_profile"] == "moss_image"
    assert captured["version"] == 1
    flattened = [entry[0] for section in tokens["qwen35_9b"] for entry in section]
    assert captured["token_count"] == len(flattened)
    mask = captured["token_mask"]
    assert all(type(flag) is bool for flag in mask)
    actual_text = moss_tokenizer.qwen35_9b.tokenizer.decode([
        token for token, editable in zip(flattened, mask) if editable
    ])
    assert actual_text == expected_user_text


def test_context_flattens_native_sections_in_token_order(moss_tokenizer):
    clip = NativeTokenizerClip(moss_tokenizer)
    first = clip.tokenize("red fox")["qwen35_9b"][0]
    second = clip.tokenize("blue bird")["qwen35_9b"][0]
    tokens = {"qwen35_9b": [first, second]}
    captured = capture_conditioning_context(clip, tokens)
    native = clip.encode_from_tokens_scheduled(tokens)
    assert native[0][0].shape[1] == len(first) + len(second)
    attached = attach_conditioning_context(native, captured)
    assert attached[0][1][CONDITIONING_CONTEXT_KEY]["token_mask"] == captured["token_mask"]
    assert captured["token_count"] == native[0][0].shape[1]
    decoded = moss_tokenizer.qwen35_9b.tokenizer.decode([
        entry[0] for entry, editable in zip(first + second, captured["token_mask"]) if editable
    ])
    assert decoded == "red foxblue bird"


def test_generic_qwen35_does_not_acquire_moss_provenance():
    from comfy.text_encoders.qwen35 import Qwen35ImageTokenizer
    clip = NativeTokenizerClip(Qwen35ImageTokenizer(model_type="qwen35_9b"))
    assert capture_conditioning_context(clip, clip.tokenize("red fox")) is None


def test_attach_context_keeps_scheduled_entries_and_never_mutates_input(moss_tokenizer):
    clip = NativeTokenizerClip(moss_tokenizer)
    tokens = clip.tokenize("red fox")
    captured = capture_conditioning_context(clip, tokens)
    native = clip.encode_from_tokens_scheduled(tokens)
    first = [native[0][0], {**native[0][1], "clip_start_percent": 0.0, "clip_end_percent": 0.5}]
    second = [native[0][0].clone(), {**native[0][1], "clip_start_percent": 0.5, "clip_end_percent": 1.0}]
    source = [first, second]
    attached = attach_conditioning_context(source, captured)
    assert len(attached) == 2
    for actual, original in zip(attached, source):
        assert CONDITIONING_CONTEXT_KEY not in original[1]
        assert actual[0] is original[0]
        assert actual[1][CONDITIONING_CONTEXT_KEY]["token_mask"] == captured["token_mask"]
        equal_metadata({k: v for k, v in actual[1].items() if k != CONDITIONING_CONTEXT_KEY}, original[1])


def test_mismatched_encoded_length_never_receives_a_valid_edit_mask(moss_tokenizer):
    clip = NativeTokenizerClip(moss_tokenizer)
    captured = capture_conditioning_context(clip, clip.tokenize("red fox"))
    mismatched = [[torch.zeros(1, captured["token_count"] + 1, 4096), {}]]
    attached = attach_conditioning_context(mismatched, captured)
    assert attached[0][1][CONDITIONING_CONTEXT_KEY]["token_mask"] is None
    assert attached[0][1][CONDITIONING_CONTEXT_KEY]["warning"]
    with pytest.raises(ValueError):
        adjust.OnekoConditioningAdjust().adjust(
            attached, "literal_detail", 1.0, True, True, "auto",
        )


def test_native_moss_condition_handling_accepts_provenance_without_changing_model_inputs(moss_tokenizer):
    from comfy.model_base import BaseModel, MOSSImage
    clip = NativeTokenizerClip(moss_tokenizer)
    tokens = clip.tokenize("red fox")
    native = clip.encode_from_tokens_scheduled(tokens)
    attached = attach_conditioning_context(native, capture_conditioning_context(clip, tokens))
    model = MOSSImage.__new__(MOSSImage)
    with patch.object(MOSSImage, "concat_cond", return_value=None), patch.object(BaseModel, "encode_adm", return_value=None):
        old = model.extra_conds(cross_attn=native[0][0], **native[0][1])
        new = model.extra_conds(cross_attn=attached[0][0], **attached[0][1])
    assert old.keys() == new.keys()
    assert CONDITIONING_CONTEXT_KEY not in new
    assert new["c_crossattn"].cond is old["c_crossattn"].cond


def test_fingerprint_binds_dtype_shape_and_content_but_allows_equivalent_clones():
    tensor = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    original = conditioning_fingerprint(tensor)
    assert conditioning_fingerprint(tensor.clone()) == original
    assert conditioning_fingerprint(tensor.reshape(2, 3, 4)) != original
    assert conditioning_fingerprint(tensor.to(torch.float16)) != original
    altered = tensor.clone()
    altered[0, 0, 0] += 1
    assert conditioning_fingerprint(altered) != original
    noncontiguous = tensor.transpose(1, 2)
    assert conditioning_fingerprint(noncontiguous) == conditioning_fingerprint(noncontiguous.contiguous())


def test_native_conditioning_average_invalidates_same_length_provenance(moss_tokenizer):
    from nodes import ConditioningAverage
    clip = NativeTokenizerClip(moss_tokenizer)
    conditionings = []
    for text in ("red fox", "blue bird"):
        tokens = clip.tokenize(text)
        conditionings.append(attach_conditioning_context(
            clip.encode_from_tokens_scheduled(tokens), capture_conditioning_context(clip, tokens),
        ))
    assert conditionings[0][0][0].shape == conditionings[1][0][0].shape
    averaged = ConditioningAverage().addWeighted(*conditionings, 0.5)[0]
    assert averaged[0][1][CONDITIONING_CONTEXT_KEY] == conditionings[0][0][1][CONDITIONING_CONTEXT_KEY]
    with pytest.raises(ValueError):
        adjust.OnekoConditioningAdjust().adjust(
            averaged, "soft_balance", 1.0, True, True, "auto",
        )


def test_chained_adjust_nodes_refresh_only_their_own_fingerprint():
    mask = [False, True, True, False, True, True, True, False]
    tensor = torch.randn((2, 8, 32), generator=torch.Generator().manual_seed(41))
    source = [[tensor, metadata(mask, tensor)]]
    first, _ = adjust.OnekoConditioningAdjust().adjust(
        source, "literal_detail", 1.0, True, True, "auto",
    )
    first_before = deepcopy(first)
    second, _ = adjust.OnekoConditioningAdjust().adjust(
        first, "negative_tamer", 0.7, True, True, "auto",
    )
    assert_adjust_metadata(first[0][1], source[0][1], first[0][0])
    assert_adjust_metadata(second[0][1], first[0][1], second[0][0])
    assert first[0][1][CONDITIONING_CONTEXT_KEY]["source_fingerprint"] != source[0][1][CONDITIONING_CONTEXT_KEY]["source_fingerprint"]
    assert second[0][1][CONDITIONING_CONTEXT_KEY]["source_fingerprint"] != first[0][1][CONDITIONING_CONTEXT_KEY]["source_fingerprint"]
    torch.testing.assert_close(first[0][0], first_before[0][0], rtol=0, atol=0)
    equal_metadata(first[0][1], first_before[0][1])
    selected = torch.tensor(mask)
    torch.testing.assert_close(second[0][0][:, ~selected], tensor[:, ~selected], rtol=0, atol=0)


def test_clone_and_native_area_metadata_keep_valid_provenance():
    from nodes import ConditioningSetAreaPercentage
    tensor = torch.randn((1, 6, 16), generator=torch.Generator().manual_seed(32))
    mask = [False, True, True, True, True, False]
    original = [[tensor, metadata(mask, tensor)]]
    clone = deepcopy(original)
    region = ConditioningSetAreaPercentage().append(clone, 0.5, 1.0, 0.0, 0.0, 0.8)[0]
    output, _ = adjust.OnekoConditioningAdjust().adjust(
        region, "soft_balance", 1.0, True, True, "auto",
    )
    assert output[0][1]["area"] == ("percentage", 1.0, 0.5, 0.0, 0.0)
    assert output[0][1]["strength"] == 0.8
    assert_adjust_metadata(output[0][1], region[0][1], output[0][0])
    assert "area" not in original[0][1]


@pytest.mark.parametrize("mode", ["neutral", "disabled", "zero-strength"])
def test_bypass_preserves_stale_provenance_and_reports_it(mode):
    tensor = torch.randn((1, 6, 16), generator=torch.Generator().manual_seed(33))
    mask = [False, True, True, True, True, False]
    meta = metadata(mask, tensor)
    changed = tensor + 1
    preset = "neutral_report_only" if mode == "neutral" else "negative_tamer"
    output, report = adjust.OnekoConditioningAdjust().adjust(
        [[changed, meta]], preset, 0.0 if mode == "zero-strength" else 1.0,
        True, mode != "disabled", "auto",
    )
    torch.testing.assert_close(output[0][0], changed, rtol=0, atol=0)
    equal_metadata(output[0][1], meta)
    assert "warning" in report.lower()


# Captured from the pre-MOSS public Adjust node with the fixed tensor below.
# This protects existing explicit profiles without duplicating their implementation.
LEGACY_RESULTS = {
    "t5/literal_detail/False": [[[-0.8789018392562866,1.656391978263855,-0.03380391746759415,0.8112940192222595],[2.8494651317596436,-1.001163363456726,1.8868080377578735,-1.9638205766677856],[12.69212818145752,-8.819952964782715,4.0872955322265625,1.9360872507095337],[1.6971512195596006e-7,3.5418807442511024e-7,-5.681766879206407e-7,7.2313400778512e-7]]],
    "t5/literal_detail/True": [[[-0.8342211842536926,1.5721861124038696,-0.03208542987704277,0.7700503468513489],[2.7046070098876953,-0.9502673149108887,1.7908885478973389,-1.8639860153198242],[12.046899795532227,-8.371573448181152,3.879509925842285,1.8376625776290894],[1.610873283652836e-7,3.36182239379923e-7,-5.392923299041286e-7,6.863721182526206e-7]]],
    "t5/soft_balance/False": [[[-1.3453868627548218,2.4623122215270996,-0.07615398615598679,1.1930789947509766],[3.690425157546997,-1.3305613994598389,2.435178518295288,-2.585808038711548],[8.643878936767578,-6.131983280181885,2.733534574508667,1.2559483051300049],[0.4423302412033081,0.9449782371520996,-1.568261742591858,1.9502743482589722]]],
    "t5/soft_balance/True": [[[-1.6585009098052979,3.035370111465454,-0.09387742727994919,1.4707461595535278],[4.5493035316467285,-1.6402250528335571,3.0019214153289795,-3.1876072883605957],[10.655582427978516,-7.559089660644531,3.3697144985198975,1.5482470989227295],[0.5452744364738464,1.1649044752120972,-1.9332457780838013,2.4041647911071777]]],
    "t5/contrast_pop/False": [[[-0.6952482461929321,1.3396246433258057,-0.016957273706793785,0.6613336801528931],[2.7421352863311768,-0.9447693228721619,1.8204091787338257,-1.8664956092834473],[13.74552059173584,-9.404829978942871,4.48538064956665,2.1703453063964844],[1.74583973944209e-7,3.583566012821393e-7,-5.605064643532387e-7,7.259018275362905e-7]]],
    "t5/contrast_pop/True": [[[-0.6171994209289551,1.1892379522323608,-0.0150536447763443,0.5870921611785889],[2.43430233001709,-0.8387091755867004,1.6160494089126587,-1.6569621562957764],[12.202444076538086,-8.349040985107422,3.9818503856658936,1.926701545715332],[1.549851162963023e-7,3.1812737688596826e-7,-4.975838123755238e-7,6.444118412218813e-7]]],
    "t5/negative_tamer/False": [[[-1.3374720811843872,2.483876943588257,-0.06368914991617203,1.2100938558578491],[3.0950300693511963,-1.1016207933425903,2.045867443084717,-2.1507835388183594],[8.67735481262207,-6.092610836029053,2.7693684101104736,1.2923718690872192],[0.277632474899292,0.5861129760742188,-0.9562896490097046,1.2030740976333618]]],
    "t5/negative_tamer/True": [[[-1.7143151760101318,3.1837284564971924,-0.08163405954837799,1.5510470867156982],[3.967078685760498,-1.412010908126831,2.6223063468933105,-2.7567834854125977],[11.12226676940918,-7.809250831604004,3.549659252166748,1.6565076112747192],[0.3558575510978699,0.7512548565864563,-1.225731611251831,1.542049527168274]]],
    "sdxl_clip/literal_detail/False": [[[-0.9352549314498901,1.8101118803024292,-0.02013269066810608,0.894989550113678],[2.9163856506347656,-1.0008549690246582,1.9370756149291992,-1.9801653623580933],[12.38444709777832,-8.44725513458252,4.0517659187316895,1.9685957431793213],[0.03749757260084152,0.07672096788883209,-0.11939603090286255,0.15516777336597443]]],
    "sdxl_clip/literal_detail/True": [[[-0.9088019728660583,1.7589143514633179,-0.019563253968954086,0.8696755170822144],[2.8338980674743652,-0.9725465774536133,1.8822870254516602,-1.924157977104187],[12.034163475036621,-8.208331108093262,3.9371650218963623,1.912915587425232],[0.036436982452869415,0.07455097883939743,-0.11601901054382324,0.15077897906303406]]],
    "sdxl_clip/soft_balance/False": [[[-1.200480341911316,2.28590989112854,-0.038350291550159454,1.1237797737121582],[3.5146610736846924,-1.2236753702163696,2.3300766944885254,-2.408259391784668],[9.72840690612793,-6.711655139923096,3.1523821353912354,1.508375644683838],[0.35063236951828003,0.7260416746139526,-1.1510051488876343,1.476860523223877]]],
    "sdxl_clip/soft_balance/True": [[[-1.3867743015289307,2.640643835067749,-0.0443015992641449,1.2981711626052856],[4.0600762367248535,-1.4135687351226807,2.691664695739746,-2.781980037689209],[11.238088607788086,-7.753188610076904,3.64157772064209,1.7424496412277222],[0.4050444960594177,0.8387109041213989,-1.329621434211731,1.7060439586639404]]],
    "sdxl_clip/contrast_pop/False": [[[-0.8348020315170288,1.6356356143951416,-0.011322839185595512,0.8121563792228699],[2.8570914268493652,-0.9699041843414307,1.900342583656311,-1.9266531467437744],[12.964954376220703,-8.767814636230469,4.271847724914551,2.0985705852508545],[1.781989595883715e-7,3.6143700299362536e-7,-5.547531145566609e-7,7.279130045390048e-7]]],
    "sdxl_clip/contrast_pop/True": [[[-0.7812470197677612,1.5307047367095947,-0.010596445761620998,0.7600541710853577],[2.673800468444824,-0.907681941986084,1.7784299850463867,-1.8030526638031006],[12.133213996887207,-8.205333709716797,3.997796058654785,1.9639410972595215],[1.6676696645845368e-7,3.382497482107283e-7,-5.191640752855164e-7,6.812152264501492e-7]]],
    "sdxl_clip/negative_tamer/False": [[[-1.191224217414856,2.2868025302886963,-0.0318819135427475,1.1274603605270386],[3.070159435272217,-1.0612578392028809,2.0373051166534424,-2.0941121578216553],[9.949360847473145,-6.825113773345947,3.239570379257202,1.5621229410171509],[0.20672966539859772,0.42549124360084534,-0.6683166027069092,0.8630144000053406]]],
    "sdxl_clip/negative_tamer/True": [[[-1.38392174243927,2.6567254066467285,-0.037039268761873245,1.3098431825637817],[3.5668015480041504,-1.23293137550354,2.366868257522583,-2.4328646659851074],[11.558812141418457,-7.929173469543457,3.7636170387268066,1.8148186206817627],[0.2401711344718933,0.49432051181793213,-0.776426374912262,1.0026192665100098]]],
    "sd15_clip/literal_detail/False": [[[-0.947364091873169,1.8444750308990479,-0.016751034185290337,0.9138619899749756],[2.9314372539520264,-1.0007387399673462,1.9483931064605713,-1.9837826490402222],[12.315234184265137,-8.365260124206543,4.043036937713623,1.9749871492385864],[0.04847504943609238,0.09876037389039993,-0.15266624093055725,0.19933101534843445]]],
    "sd15_clip/literal_detail/True": [[[-0.9253873229026794,1.801687240600586,-0.016362447291612625,0.8926623463630676],[2.863434314727783,-0.9775237441062927,1.9031946659088135,-1.9377632141113281],[12.029547691345215,-8.171204566955566,3.949247360229492,1.9291716814041138],[0.04735053330659866,0.09646935015916824,-0.14912471175193787,0.1947069615125656]]],
    "sd15_clip/soft_balance/False": [[[-1.1701537370681763,2.248016834259033,-0.0307635348290205,1.1086267232894897],[3.4820632934570312,-1.2028518915176392,2.3108344078063965,-2.3740806579589844],[9.949151992797852,-6.821436882019043,3.2409162521362305,1.5638573169708252],[0.3313811421394348,0.681678295135498,-1.069807529449463,1.382272720336914]]],
    "sd15_clip/soft_balance/True": [[[-1.3334240913391113,2.561680316925049,-0.035055939108133316,1.2633123397827148],[3.967911958694458,-1.3706846237182617,2.633262872695923,-2.705333709716797],[11.337347030639648,-7.7732253074646,3.693118095397949,1.7820607423782349],[0.3776184320449829,0.7767921686172485,-1.2190767526626587,1.5751398801803589]]],
    "sd15_clip/contrast_pop/False": [[[-0.8652768731117249,1.701675534248352,-0.009626071900129318,0.8460246920585632],[2.88287615776062,-0.9754273891448975,1.9183002710342407,-1.9400032758712769],[12.790403366088867,-8.627330780029297,4.223309516906738,2.081536054611206],[1.7899799331644317e-7,3.6211613974046486e-7,-5.534745923796436e-7,7.283524610102177e-7]]],
    "sd15_clip/contrast_pop/True": [[[-0.8195195198059082,1.6116878986358643,-0.00911702774465084,0.8012853860855103],[2.730424642562866,-0.9238450527191162,1.816857099533081,-1.8374124765396118],[12.11402416229248,-8.171102523803711,3.9999732971191406,1.9714607000350952],[1.6953225667748484e-7,3.429667856380547e-7,-5.242058591647947e-7,6.898358719809039e-7]]],
    "sd15_clip/negative_tamer/False": [[[-1.1579324007034302,2.239423990249634,-0.025480173528194427,1.1069719791412354],[3.0615453720092773,-1.0513620376586914,2.033318519592285,-2.0795888900756836],[10.278319358825684,-7.014331340789795,3.3612589836120605,1.6319938898086548],[0.1892172396183014,0.38735049962997437,-0.6033157706260681,0.7836169600486755]]],
    "sd15_clip/negative_tamer/True": [[[-1.312116026878357,2.5376129150390625,-0.028872966766357422,1.2543699741363525],[3.469203233718872,-1.1913553476333618,2.304063558578491,-2.3564951419830322],[11.646921157836914,-7.948319435119629,3.8088250160217285,1.8493008613586426],[0.21441231667995453,0.43892785906791687,-0.6836498379707336,0.8879588842391968]]],
}


@pytest.mark.parametrize("case,expected", list(LEGACY_RESULTS.items()))
def test_existing_explicit_profiles_keep_their_numeric_results(case, expected):
    profile, preset, preserve = case.split("/")
    tensor = torch.tensor([[[-1.0, 2.0, 0.0, 1.0], [3.0, -1.0, 2.0, -2.0],
                            [12.0, -8.0, 4.0, 2.0], [0.1, 0.2, -0.3, 0.4]]])
    output, _ = adjust.OnekoConditioningAdjust().adjust(
        [[tensor, {}]], preset, 1.0, preserve == "True", True, profile,
    )
    torch.testing.assert_close(output[0][0], torch.tensor(expected), rtol=1e-6, atol=1e-7)
