from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from comfy.quant_ops import TensorWiseINT8Layout
from Oneko_ComfyUI_Nodes.nodes import embedding_sculpt_core as core
from tests.test_embedding_sculpt_core import dense_neighbors


class AcceleratorRows:
    """CPU-backed stand-in permitting only accelerator views and CPU copies."""
    device = torch.device("cuda")

    def __init__(self, data):
        self.data = data
        self.shape = data.shape
        self.ndim = data.ndim

    def detach(self):
        return self

    def __getitem__(self, index):
        assert isinstance(index, slice), "Source-device gather is forbidden"
        return AcceleratorRows(self.data[index])

    def to(self, *, device, copy=False):
        assert torch.device(device).type == "cpu"
        return self.data.clone()


class QuantizedAcceleratorWeight(SimpleNamespace):
    device = torch.device("cuda")


@pytest.mark.parametrize("quantized", [False, True], ids=["dense", "int8-row-scales"])
def test_cpu_fallback_reads_accelerator_rows_without_accelerator_allocations(quantized):
    generator = torch.Generator().manual_seed(191)
    shape = (41, 256)
    if quantized:
        data = torch.randint(-100, 100, shape, dtype=torch.int8, generator=generator)
        scale = torch.pow(2.0, torch.arange(shape[0]) % 5 - 10).reshape(-1, 1)
        params = TensorWiseINT8Layout.Params(
            scale=scale, orig_dtype=torch.bfloat16, orig_shape=shape,
        )
        dense = TensorWiseINT8Layout.dequantize_embedding(data, params, torch.arange(shape[0])).float()
        weight = QuantizedAcceleratorWeight(
            shape=shape, layout_cls=TensorWiseINT8Layout,
            _qdata=AcceleratorRows(data),
            _params=core.replace(params, scale=AcceleratorRows(scale)),
        )
        originals = [(data, data.clone()), (scale, scale.clone())]
    else:
        data = torch.randn(*shape, generator=generator)
        dense = data.clone()
        weight = AcceleratorRows(data)
        originals = [(data, data.clone())]
    token_ids = [1, 9, 29, 37]
    expected = dense_neighbors(dense, token_ids, 6)
    real_tensor = torch.tensor
    real_search = core._search_on_device

    def cpu_tensor_only(*args, **kwargs):
        device = torch.device(kwargs.get("device") or "cpu")
        assert device.type == "cpu", "CPU fallback attempted an accelerator allocation"
        return real_tensor(*args, **kwargs)

    def search_with_initial_oom(weight, ids, top_k, device, **kwargs):
        if torch.device(device).type != "cpu":
            raise torch.cuda.OutOfMemoryError("simulated exhausted accelerator")
        return real_search(weight, ids, top_k, device, **kwargs)

    with (
        patch.object(core, "QuantizedTensor", QuantizedAcceleratorWeight),
        patch.object(core.torch, "tensor", side_effect=cpu_tensor_only),
        patch.object(core, "_search_on_device", side_effect=search_with_initial_oom),
        patch.object(core.model_management, "soft_empty_cache"),
        patch.object(core, "CPU_WORK_BYTES", 4096),
    ):
        for row_ids in ([35, 1, 9, 35], range(5, 11)):
            actual = core._embedding_rows(weight, row_ids, torch.device("cpu"))
            torch.testing.assert_close(actual, dense[list(row_ids)], rtol=0, atol=0)
        actual, stats = core.chunked_top_neighbors(weight, token_ids, 6, preferred_device="cuda")
        assert stats["cpu_fallback"]
        assert stats["chunk_rows"] < shape[0]
        for token_id in token_ids:
            assert actual[token_id][0] == expected[token_id][0]
            torch.testing.assert_close(
                real_tensor(actual[token_id][1]), real_tensor(expected[token_id][1]), rtol=1e-6, atol=1e-6,
            )
        mean, _ = core.streaming_mean_magnitude(weight, preferred_device="cpu")
        assert mean == pytest.approx(torch.linalg.vector_norm(dense, dim=1).mean().item(), rel=1e-6)
    for source, original in originals:
        torch.testing.assert_close(source, original, rtol=0, atol=0)


def test_contiguous_cpu_rows_do_not_alias_the_embedding_table():
    weight = torch.randn(11, 7, generator=torch.Generator().manual_seed(71))
    original = weight.clone()
    rows = core._embedding_rows(weight, range(2, 8), torch.device("cpu"))
    rows.zero_()
    torch.testing.assert_close(weight, original, rtol=0, atol=0)
