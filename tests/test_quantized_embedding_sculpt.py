"""Quantized embedding access must not expand the complete vocabulary table."""
from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch

from comfy.quant_ops import QuantizedTensor, TensorCoreFP8E4M3Layout, TensorWiseINT8Layout
from Oneko_ComfyUI_Nodes.nodes import embedding_sculpt_core as core
from tests.test_embedding_sculpt_core import dense_neighbors, dense_refine


@pytest.fixture(params=["int8-row", "int8-convrot", "int8-convrot-fp16", "int8-scalar", "fp8"])
def quantized_embedding(request):
    shape = (41, 256)
    generator = torch.Generator().manual_seed(191)
    if request.param == "fp8":
        data = (torch.randn(*shape, generator=generator) * 8).to(torch.float8_e4m3fn)
        layout = TensorCoreFP8E4M3Layout
        params = layout.Params(
            scale=torch.tensor(0.03), orig_dtype=torch.bfloat16, orig_shape=shape,
        )
        weight = QuantizedTensor(data, "TensorCoreFP8E4M3Layout", params)
    else:
        data = torch.randint(-100, 100, shape, dtype=torch.int8, generator=generator)
        # Dyadic row scales avoid native CPU GEMM rounding changing BF16
        # half-way values when ConvRot is decoded in different batch sizes.
        scale = (
            torch.tensor(0.013)
            if request.param == "int8-scalar"
            else torch.pow(2.0, torch.arange(shape[0]) % 5 - 10).reshape(-1, 1)
        )
        layout = TensorWiseINT8Layout
        params = layout.Params(
            scale=scale,
            orig_dtype=torch.float16 if request.param.endswith("fp16") else torch.bfloat16,
            orig_shape=shape, convrot=request.param.startswith("int8-convrot"),
            convrot_groupsize=256,
        )
        weight = QuantizedTensor(data, "TensorWiseINT8Layout", params)
    # The dense reference is made once, before forbidding the expensive fallback.
    return weight, weight.dequantize().float()


@contextmanager
def forbid_full_dequantization(weight):
    with patch.object(
        QuantizedTensor, "dequantize",
        side_effect=AssertionError("full vocabulary dequantization is forbidden"),
    ):
        if weight.layout_cls is TensorWiseINT8Layout:
            with patch.object(
                TensorWiseINT8Layout, "dequantize",
                side_effect=AssertionError("INT8 must use native row-aware decoding"),
            ):
                yield
        else:
            yield


@pytest.mark.parametrize("row_ids", [[35, 1, 9, 35], range(5, 11), []])
def test_quantized_row_access_preserves_order_and_source(quantized_embedding, row_ids):
    weight, dense = quantized_embedding
    original_data = weight._qdata.clone()
    original_scale = weight.params.scale.clone()
    with forbid_full_dequantization(weight):
        actual = core._embedding_rows(weight, row_ids, torch.device("cpu"))
    torch.testing.assert_close(actual, dense[list(row_ids)], rtol=0, atol=0)
    assert actual.dtype == torch.float32
    assert actual.device.type == "cpu"
    # CPU float8 does not implement every equality operator; compare its values as float.
    torch.testing.assert_close(weight._qdata.float(), original_data.float(), rtol=0, atol=0)
    torch.testing.assert_close(weight.params.scale, original_scale, rtol=0, atol=0)


def test_quantized_chunked_search_matches_dense(quantized_embedding):
    weight, dense = quantized_embedding
    token_ids = [1, 9, 29, 37]
    expected = dense_neighbors(dense, token_ids, 6)
    with forbid_full_dequantization(weight):
        actual, stats = core._search_on_device(
            weight, token_ids, 6, torch.device("cpu"),
            work_bytes=5 * 256 * 4, query_batch_size=2,
        )
    assert stats["query_batches"] == 2
    assert stats["chunk_rows"] < weight.shape[0]
    for token_id in token_ids:
        assert actual[token_id][0] == expected[token_id][0]
        torch.testing.assert_close(
            torch.tensor(actual[token_id][1]), torch.tensor(expected[token_id][1]),
            rtol=1e-6, atol=1e-6,
        )


def test_quantized_streaming_mean_matches_dense(quantized_embedding):
    weight, dense = quantized_embedding
    expected = torch.linalg.vector_norm(dense, dim=1).mean().item()
    with forbid_full_dequantization(weight), patch.object(core, "CPU_WORK_BYTES", 2048):
        actual, stats = core.streaming_mean_magnitude(weight, preferred_device=torch.device("cpu"))
    assert stats["chunk_rows"] < weight.shape[0]
    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("method", [
    "forward", "backward", "maximum_absolute", "add_minimum_absolute",
])
def test_quantized_session_original_and_refinement_match_dense(quantized_embedding, method):
    weight, dense = quantized_embedding
    expected, expected_count = dense_refine(dense, 9, method, 0.35, 8)
    with forbid_full_dequantization(weight), patch.object(core, "CPU_WORK_BYTES", 2048):
        session = core.EmbeddingSculptSession(
            weight, [9, 9], 8, preferred_device=torch.device("cpu"),
        )
        original = session.original(9)
        actual, count = session.sculpt(9, method, 0.35)
        repeated, repeated_count = session.sculpt(9, method, 0.35)
    torch.testing.assert_close(original, dense[9], rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(repeated, actual, rtol=0, atol=0)
    assert count == expected_count == repeated_count
    assert session.cache_entries == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_convrot_arbitrary_scales_match_native_embedding_lookup(dtype):
    shape = (41, 256)
    data = torch.randint(-100, 100, shape, dtype=torch.int8, generator=torch.Generator().manual_seed(191))
    params = TensorWiseINT8Layout.Params(
        scale=torch.linspace(0.002, 0.02, shape[0]).reshape(-1, 1),
        orig_dtype=dtype, orig_shape=shape, convrot=True, convrot_groupsize=256,
    )
    weight = QuantizedTensor(data, "TensorWiseINT8Layout", params)
    row_ids = [29, 37, 1, 9, 29]
    # Match the native embedding operation, including its requested batch shape.
    expected = TensorWiseINT8Layout.dequantize_embedding(data, params, torch.tensor(row_ids)).float()
    with forbid_full_dequantization(weight):
        actual = core._embedding_rows(weight, row_ids, torch.device("cpu"))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
