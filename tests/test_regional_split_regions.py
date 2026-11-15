from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import comfy.samplers
from comfy.conds import CONDRegular
from nodes import ConditioningSetMask

from Oneko_ComfyUI_Nodes.nodes.regional_split_regions import (
    OnekoNativeRegionalRectConditioning,
    OnekoNativeRegionalSplitConditioning,
    OnekoRegionalRectMasks,
    OnekoSplitMasks,
)


def conditioning(value, length):
    return [[torch.full((1, length, 4), float(value)), {
        "pooled_output": torch.tensor([[value]], dtype=torch.float32),
        "attention_mask": torch.ones((1, length), dtype=torch.int64),
        "marker": {"region": value},
    }]]


def rect_arguments(region_count=2, set_cond_area="default", **overrides):
    args = dict(
        conditioning_1=conditioning(2, 5), conditioning_2=conditioning(6, 7),
        conditioning_3=conditioning(9, 11) if region_count == 3 else None,
        width=64, height=48, region_count=region_count,
        x_1=0.0, y_1=0.0, w_1=0.25, h_1=1.0,
        x_2=0.75, y_2=0.0, w_2=0.25, h_2=1.0,
        x_3=0.375, y_3=0.25, w_3=0.25, h_3=0.5,
        soft_edge=0, strength_1=0.75, strength_2=1.5, strength_3=0.25,
        set_cond_area=set_cond_area,
    )
    args.update(overrides)
    return args


def assert_same_conditioning(actual, expected):
    assert len(actual) == len(expected)
    for (actual_tensor, actual_meta), (expected_tensor, expected_meta) in zip(actual, expected):
        assert actual_tensor is expected_tensor
        assert actual_meta.keys() == expected_meta.keys()
        for key, expected_value in expected_meta.items():
            if isinstance(expected_value, torch.Tensor):
                torch.testing.assert_close(actual_meta[key], expected_value, rtol=0, atol=0)
            else:
                assert actual_meta[key] == expected_value


@pytest.mark.parametrize("region_count", [2, 3])
@pytest.mark.parametrize("set_cond_area", ["default", "mask bounds"])
def test_rect_apply_matches_native_set_mask_and_preserves_inputs(region_count, set_cond_area):
    args = rect_arguments(region_count, set_cond_area)
    # Multiple conditioning entries and distinct text lengths must remain intact.
    args["conditioning_1"] += conditioning(3, 9)
    original = [entry for i in range(1, region_count + 1) for entry in args[f"conditioning_{i}"]]
    old_tensors = [entry[0].clone() for entry in original]
    old_meta = [dict(entry[1]) for entry in original]
    result, *masks = OnekoNativeRegionalRectConditioning().apply(**args)
    mask_args = {k: v for k, v in args.items()
                 if not k.startswith(("conditioning_", "strength_")) and k != "set_cond_area"}
    public_masks = OnekoRegionalRectMasks().build(**mask_args)
    expected = []
    for i in range(region_count):
        expected += ConditioningSetMask().append(
            args[f"conditioning_{i + 1}"], public_masks[i], set_cond_area, args[f"strength_{i + 1}"],
        )[0]
    assert_same_conditioning(result, expected)
    for mask, public_mask in zip(masks, public_masks):
        torch.testing.assert_close(mask, public_mask, rtol=0, atol=0)
        assert mask.shape == (1, 48, 64)
        assert mask.dtype == torch.float32
    for old, tensor, meta, new in zip(original, old_tensors, old_meta, result):
        torch.testing.assert_close(old[0], tensor, rtol=0, atol=0)
        assert old[1].keys() == meta.keys()
        assert new[1] is not old[1]
        assert new[1]["marker"] is old[1]["marker"]
        for key in meta:
            assert old[1][key] is meta[key]
    if region_count == 2:
        assert torch.count_nonzero(masks[2]) == 0


def test_rect_gaps_are_not_filled_and_overlaps_are_not_renormalized():
    _, first, second, unused = OnekoNativeRegionalRectConditioning().apply(**rect_arguments())
    assert torch.all(first[:, :, :16] == 1)
    assert torch.all(second[:, :, 48:] == 1)
    assert torch.count_nonzero((first + second)[:, :, 16:48]) == 0
    assert torch.count_nonzero(unused) == 0

    _, first, second, _ = OnekoNativeRegionalRectConditioning().apply(
        **rect_arguments(w_1=0.75, x_2=0.25, w_2=0.75),
    )
    assert torch.all(first[:, :, 16:48] == 1)
    assert torch.all(second[:, :, 16:48] == 1)
    assert torch.all((first + second)[:, :, 16:48] == 2)


def test_rect_soft_edges_retain_fractional_mask_and_true_empty_area():
    _, mask, _, _ = OnekoNativeRegionalRectConditioning().apply(
        **rect_arguments(x_1=0.25, y_1=0.25, w_1=0.5, h_1=0.5, soft_edge=3),
    )
    assert torch.isfinite(mask).all()
    assert mask.min() == 0 and mask.max() == 1
    assert ((mask > 0) & (mask < 1)).any()
    assert mask[0, 24, 32] == 1
    assert mask[0, 0, 0] == 0
    assert 0 < mask[0, 24, 15] < 1  # feather extends beyond the hard rectangle


@pytest.mark.parametrize("rect,expected_area", [
    ((0.75, -0.5, 2.0, 2.0), 16 * 48),
    ((1.0, 0.0, 0.5, 1.0), 0),
    ((0.25, 0.25, 0.0, 0.5), 0),
    ((0.25, 0.25, 0.5, -1.0), 0),
])
def test_rect_coordinates_clip_and_degenerate_regions_remain_empty(rect, expected_area):
    values = dict(zip(("x_1", "y_1", "w_1", "h_1"), rect))
    _, mask, _, _ = OnekoNativeRegionalRectConditioning().apply(**rect_arguments(**values))
    assert mask.sum().item() == expected_area
    assert torch.isfinite(mask).all()


@pytest.mark.parametrize("node_type", [OnekoNativeRegionalRectConditioning, OnekoNativeRegionalSplitConditioning])
def test_third_conditioning_is_required_for_three_regions(node_type):
    if node_type is OnekoNativeRegionalRectConditioning:
        args = rect_arguments(3, conditioning_3=None)
    else:
        args = split_arguments(3, conditioning_3=None)
    with pytest.raises(ValueError, match="conditioning_3 is required"):
        node_type().apply(**args)


def split_arguments(region_count=2, **overrides):
    args = dict(
        conditioning_1=conditioning(2, 5), conditioning_2=conditioning(6, 7),
        conditioning_3=conditioning(9, 11) if region_count == 3 else None,
        width=64, height=48, region_count=region_count, orientation="horizontal",
        split_1=0.25, split_2=0.75, overlap=0.0,
        strength_1=0.75, strength_2=1.5, strength_3=0.25, set_cond_area="default",
    )
    args.update(overrides)
    return args


@pytest.mark.parametrize("region_count", [2, 3])
@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
@pytest.mark.parametrize("set_cond_area", ["default", "mask bounds"])
@pytest.mark.parametrize("overlap", [0.0, 0.125])
def test_split_apply_preserves_native_mask_contract(region_count, orientation, set_cond_area, overlap):
    args = split_arguments(region_count, orientation=orientation, set_cond_area=set_cond_area, overlap=overlap)
    result, *masks = OnekoNativeRegionalSplitConditioning().apply(**args)
    expected_masks = OnekoSplitMasks().build(64, 48, region_count, orientation, 0.25, 0.75, overlap)
    expected = []
    for i in range(region_count):
        expected += ConditioningSetMask().append(
            args[f"conditioning_{i + 1}"], expected_masks[i], set_cond_area, args[f"strength_{i + 1}"],
        )[0]
    assert_same_conditioning(result, expected)
    for mask, expected_mask in zip(masks, expected_masks):
        torch.testing.assert_close(mask, expected_mask, rtol=0, atol=0)
    coverage = sum(masks)
    if overlap == 0:
        assert torch.all(coverage == 1)
    else:
        assert torch.all(coverage >= 1)
        assert torch.any(coverage > 1)
    axis = 2 if orientation == "horizontal" else 1
    axis_size = 64 if orientation == "horizontal" else 48
    assert torch.all(masks[0].narrow(axis, 0, axis_size // 4) == 1)
    assert all("mask" not in args[f"conditioning_{i + 1}"][0][1] for i in range(region_count))


@pytest.mark.parametrize("splits", [(0.75, 0.25), (0.5, 0.5)])
def test_three_way_split_handles_reversed_or_equal_boundaries(splits):
    result, *masks = OnekoNativeRegionalSplitConditioning().apply(
        **split_arguments(3, split_1=splits[0], split_2=splits[1]),
    )
    assert len(result) == 3
    assert torch.all(sum(masks) == 1)
    assert all(torch.count_nonzero(mask) > 0 for mask in masks)


def core_conditions(conditionings, dims):
    # MOSS uses CONDRegular rather than repeat-padding contexts to a common length.
    result = [{**meta, "model_conds": {"c_crossattn": CONDRegular(tensor)}, "uuid": str(i)}
              for i, (tensor, meta) in enumerate(conditionings)]
    comfy.samplers.resolve_areas_and_cond_masks_multidim(result, dims, torch.device("cpu"))
    return result


@pytest.mark.parametrize("set_cond_area", ["default", "mask bounds"])
def test_core_mask_resolution_handles_moss_latents_and_unequal_contexts(set_cond_area):
    args = rect_arguments(
        set_cond_area=set_cond_area, width=512, height=256,
        x_1=0.125, y_1=0.25, w_1=0.5, h_1=0.5,
        x_2=0.625, y_2=0.25, w_2=0.25, h_2=0.5,
    )
    conditioned, *masks = OnekoNativeRegionalRectConditioning().apply(**args)
    dims = (256 // 16, 512 // 16)
    latent = torch.ones((2, 128, *dims), dtype=torch.float32)
    native = core_conditions(conditioned, dims)
    areas = []
    for i, data in enumerate(native):
        item = comfy.samplers.get_area_and_mult(data, latent, torch.tensor([1.0]))
        areas.append(item)
        expected_mask = F.interpolate(masks[i].unsqueeze(1), size=dims, mode="bilinear").squeeze(1)
        torch.testing.assert_close(data["mask"], expected_mask, rtol=0, atol=0)
        if set_cond_area == "mask bounds":
            height, width, y, x = item.area
            expected_mask = expected_mask[:, y:y + height, x:x + width]
        else:
            assert item.area is None
        expected_mult = expected_mask.unsqueeze(1).repeat(2, 128, 1, 1) * args[f"strength_{i + 1}"]
        torch.testing.assert_close(item.mult, expected_mult, rtol=0, atol=0)
        assert item.input_x.shape == item.mult.shape
        assert item.conditioning["c_crossattn"].cond.shape == (2, (5, 7)[i], 4)
        assert conditioned[i][1]["mask"].shape == (1, 256, 512)
    if set_cond_area == "mask bounds":
        assert tuple(areas[0].area) == (8, 16, 4, 4)
        assert tuple(areas[1].area) == (8, 8, 4, 20)
    assert not comfy.samplers.can_concat_cond(*areas)


class ConstantRegionModel:
    def __init__(self):
        self.context_lengths = []
        self.current_patcher = SimpleNamespace(
            prepare_state=lambda *args: None,
            get_free_memory=lambda device: 1e12,
            apply_hooks=lambda **kwargs: {},
        )

    def memory_required(self, input_shape, cond_shapes):
        return 0

    def apply_model(self, latent, timestep, c_crossattn, **kwargs):
        self.context_lengths.append(c_crossattn.shape[1])
        return c_crossattn[:, :1, :1].reshape(-1, 1, 1, 1).expand_as(latent)


@pytest.mark.parametrize("set_cond_area", ["default", "mask bounds"])
@pytest.mark.parametrize("soft_edge", [0, 8])
def test_core_sampling_weights_overlaps_and_leaves_gaps_unassigned(set_cond_area, soft_edge):
    conditioned, *masks = OnekoNativeRegionalRectConditioning().apply(**rect_arguments(
        width=512, height=256, set_cond_area=set_cond_area, soft_edge=soft_edge,
        x_1=0.0, w_1=0.5, x_2=0.25, w_2=0.5, strength_1=1.0, strength_2=3.0,
    ))
    dims = (16, 32)
    latent = torch.zeros((1, 128, *dims), dtype=torch.float32)
    conditions = core_conditions(conditioned, dims)
    model = ConstantRegionModel()
    output = comfy.samplers.calc_cond_batch(model, [conditions], latent, torch.tensor([1.0]), {})[0]
    weights = [F.interpolate(m.unsqueeze(1), size=dims, mode="bilinear") for m in masks[:2]]
    numerator = 2 * weights[0] + 6 * weights[1] * 3
    denominator = weights[0] + weights[1] * 3
    expected = numerator / denominator.clamp_min(1e-37)
    torch.testing.assert_close(output, expected.expand_as(output), rtol=1e-6, atol=1e-6)
    assert torch.isfinite(output).all()
    assert torch.all(output[..., -1] == 0)  # native gap: no implicit background region
    assert torch.all(output[..., 0] == 2)
    assert set(model.context_lengths) == {5, 7}
    if soft_edge == 0:
        assert torch.all(output[..., 8:16] == 5)  # (2 * 1 + 6 * 3) / (1 + 3)
        assert torch.all(output[..., 16:24] == 6)


@pytest.mark.parametrize("set_cond_area", ["default", "mask bounds"])
@pytest.mark.parametrize("inactive_reason", ["empty masks", "zero strength"])
def test_core_sampling_inactive_regions_produce_finite_zero(set_cond_area, inactive_reason):
    overrides = {"w_1": 0.0, "w_2": 0.0} if inactive_reason == "empty masks" else {
        "strength_1": 0.0, "strength_2": 0.0,
    }
    conditioned, *_ = OnekoNativeRegionalRectConditioning().apply(**rect_arguments(
        width=512, height=256, set_cond_area=set_cond_area, **overrides,
    ))
    latent = torch.zeros((1, 128, 16, 32), dtype=torch.float32)
    result = comfy.samplers.calc_cond_batch(
        ConstantRegionModel(), [core_conditions(conditioned, (16, 32))], latent, torch.tensor([1.0]), {},
    )[0]
    assert torch.isfinite(result).all()
    assert torch.count_nonzero(result) == 0
