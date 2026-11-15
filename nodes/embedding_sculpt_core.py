import logging
import time
from dataclasses import replace

import torch
import comfy.utils
from comfy.quant_ops import QuantizedTensor
import comfy.model_management as model_management


GPU_WORK_BYTES = 32 * 1024 * 1024
CPU_WORK_BYTES = 64 * 1024 * 1024
MAX_QUERY_BATCH = 256


def maximum_absolute_values(tensors, reversed=False):
    shape = tensors.shape
    tensors = tensors.reshape(shape[0], -1)
    tensors_abs = torch.abs(tensors)
    idx = torch.argmin(tensors_abs, dim=0) if reversed else torch.argmax(tensors_abs, dim=0)
    columns = torch.arange(tensors.shape[1], device=tensors.device)
    return tensors[idx, columns].reshape(shape[1:])


def _as_device(device):
    return device if isinstance(device, torch.device) else torch.device(device)


def _gather_embedding_rows(source, row_ids, device):
    # Views avoid source-device allocation; copies protect tables from normalization.
    if isinstance(row_ids, range) and row_ids.step == 1:
        return source[row_ids.start:row_ids.stop].to(device=device, copy=True)
    if device.type == "cpu" and source.device.type != "cpu":
        # An OOM fallback must allocate neither indices nor gathered rows on the GPU.
        return torch.cat(
            [source[index:index + 1].to(device=device, copy=True) for index in row_ids],
            dim=0,
        )
    indices = torch.tensor(row_ids, device=source.device, dtype=torch.long)
    return torch.index_select(source, 0, indices).to(device=device)


def _embedding_rows(weight, row_ids, device):
    model_management.throw_exception_if_processing_interrupted()
    device = _as_device(device)
    if not row_ids:
        return torch.empty((0, int(weight.shape[1])), device=device, dtype=torch.float32)
    if isinstance(weight, QuantizedTensor):
        # Indexing the wrapper falls back to decoding the ENTIRE embedding table.
        # Gather stored rows and per-row scales first, as Core Embedding.host_rows does.
        layout = weight.layout_cls
        params = weight._params
        rows = _gather_embedding_rows(weight._qdata, row_ids, device)
        scale = params.scale
        if scale.ndim >= 2:
            scale = _gather_embedding_rows(scale, row_ids, device)
        params = replace(params, scale=scale.to(device), orig_shape=(len(row_ids), int(weight.shape[1])))
        if hasattr(layout, "dequantize_embedding"):
            rows = layout.dequantize_embedding(rows, params, torch.arange(len(row_ids), device=device))
        elif rows.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and scale.numel() == 1:
            rows = layout.dequantize(rows, params)
        else:
            raise RuntimeError(f"Oneko Sculpt: unsupported quantized embedding layout {layout.__name__}.")
    else:
        rows = _gather_embedding_rows(weight.detach(), row_ids, device)
    return rows.to(device=device, dtype=torch.float32)


def _normalized_in_place(rows):
    norms = torch.linalg.vector_norm(rows, dim=1, keepdim=True).clamp_min_(1e-12)
    rows.div_(norms)
    return rows


def _chunk_rows(hidden_size, query_count, work_bytes, vocab_size):
    bytes_per_row = max(1, (int(hidden_size) + int(query_count)) * 4)
    return max(1, min(int(vocab_size), int(work_bytes) // bytes_per_row))


def _search_on_device(
    weight,
    token_ids,
    top_k,
    device,
    work_bytes=None,
    query_batch_size=MAX_QUERY_BATCH,
):
    device = _as_device(device)
    vocab_size, hidden_size = (int(weight.shape[0]), int(weight.shape[1]))
    requested_k = max(1, int(top_k))
    k = min(requested_k, max(1, vocab_size - 1))
    token_ids = list(dict.fromkeys(int(token_id) for token_id in token_ids))
    budget = int(work_bytes or (CPU_WORK_BYTES if device.type == "cpu" else GPU_WORK_BYTES))
    query_batch_size = max(1, int(query_batch_size))
    result = {}
    max_chunk_rows = 0
    query_batches = 0

    progress = comfy.utils.ProgressBar(vocab_size * ((len(token_ids) + query_batch_size - 1) // query_batch_size))
    progress.update_absolute(0)
    started = time.perf_counter()
    logging.info("Oneko Sculpt: searching %d tokens in %d x %d embedding table on %s", len(token_ids), vocab_size, hidden_size, device)
    with torch.no_grad():
        for query_start in range(0, len(token_ids), query_batch_size):
            query_ids = token_ids[query_start : query_start + query_batch_size]
            query_batches += 1
            queries = _normalized_in_place(_embedding_rows(weight, query_ids, device))
            chunk_rows = _chunk_rows(hidden_size, len(query_ids), budget, vocab_size)
            max_chunk_rows = max(max_chunk_rows, chunk_rows)
            best_scores = None
            best_ids = None

            for row_start in range(0, vocab_size, chunk_rows):
                row_end = min(vocab_size, row_start + chunk_rows)
                rows = _embedding_rows(weight, range(row_start, row_end), device)
                _normalized_in_place(rows)
                scores = torch.matmul(rows, queries.T)

                for query_column, token_id in enumerate(query_ids):
                    if row_start <= token_id < row_end:
                        scores[token_id - row_start, query_column] = -torch.inf

                local_k = min(k, row_end - row_start)
                local_scores, local_offsets = torch.topk(scores, k=local_k, dim=0, largest=True)
                local_ids = local_offsets + row_start

                if best_scores is None:
                    best_scores, best_ids = local_scores, local_ids
                else:
                    merged_scores = torch.cat((best_scores, local_scores), dim=0)
                    merged_ids = torch.cat((best_ids, local_ids), dim=0)
                    merge_k = min(k, merged_scores.shape[0])
                    best_scores, positions = torch.topk(merged_scores, k=merge_k, dim=0, largest=True)
                    best_ids = torch.gather(merged_ids, 0, positions)

                del rows, scores, local_scores, local_offsets, local_ids
                progress.update(row_end - row_start)

            best_scores = best_scores.detach().cpu()
            best_ids = best_ids.detach().cpu()
            for query_column, token_id in enumerate(query_ids):
                result[token_id] = (
                    [int(value) for value in best_ids[:, query_column].tolist()],
                    [float(value) for value in best_scores[:, query_column].tolist()],
                )
            del queries, best_scores, best_ids

    elapsed = time.perf_counter() - started
    logging.info("Oneko Sculpt: embedding search finished in %.2fs", elapsed)
    return result, {
        "elapsed_seconds": elapsed,
        "device": str(device),
        "cpu_fallback": False,
        "chunk_rows": max_chunk_rows,
        "query_batches": query_batches,
        "unique_tokens": len(token_ids),
    }


def chunked_top_neighbors(weight, token_ids, top_k, preferred_device=None):
    token_ids = list(dict.fromkeys(int(token_id) for token_id in token_ids))
    if not token_ids:
        return {}, {
            "device": "disabled",
            "cpu_fallback": False,
            "chunk_rows": 0,
            "query_batches": 0,
            "unique_tokens": 0,
        }

    preferred = _as_device(preferred_device or model_management.get_torch_device())
    try:
        return _search_on_device(weight, token_ids, top_k, preferred)
    except Exception as error:
        if preferred.type == "cpu" or not model_management.is_oom(error):
            raise
        model_management.soft_empty_cache()
        neighbors, stats = _search_on_device(weight, token_ids, top_k, torch.device("cpu"))
        stats["cpu_fallback"] = True
        return neighbors, stats


def streaming_mean_magnitude(weight, preferred_device=None):
    preferred = _as_device(preferred_device or model_management.get_torch_device())

    def calculate(device):
        device = _as_device(device)
        vocab_size, hidden_size = (int(weight.shape[0]), int(weight.shape[1]))
        budget = CPU_WORK_BYTES if device.type == "cpu" else GPU_WORK_BYTES
        chunk_rows = _chunk_rows(hidden_size, 1, budget, vocab_size)
        total = 0.0
        count = 0
        progress = comfy.utils.ProgressBar(vocab_size)
        progress.update_absolute(0)
        with torch.no_grad():
            for start in range(0, vocab_size, chunk_rows):
                end = min(vocab_size, start + chunk_rows)
                rows = _embedding_rows(weight, range(start, end), device)
                total += torch.linalg.vector_norm(rows, dim=1).sum().item()
                count += rows.shape[0]
                del rows
                progress.update(end - start)
        return total / max(1, count), chunk_rows, str(device)

    try:
        mean, chunk_rows, device = calculate(preferred)
        return mean, {"device": device, "cpu_fallback": False, "chunk_rows": chunk_rows}
    except Exception as error:
        if preferred.type == "cpu" or not model_management.is_oom(error):
            raise
        model_management.soft_empty_cache()
        mean, chunk_rows, device = calculate(torch.device("cpu"))
        return mean, {"device": device, "cpu_fallback": True, "chunk_rows": chunk_rows}


def refine_from_neighbors(weight, token_id, neighbor_ids, neighbor_scores, method, intensity):
    row_ids = [int(token_id), *[int(value) for value in neighbor_ids]]
    rows = _embedding_rows(weight, row_ids, torch.device("cpu"))
    initial_weight = rows[0]
    candidate_weights = [rows[index] for index in range(1, rows.shape[0])]
    pre_mag = torch.norm(initial_weight)
    if pre_mag == 0:
        return initial_weight, 0

    previous_cos_score = 0.0
    cos_score = 1.0
    selected_scores = []
    selected_weights = []
    initial_clone = initial_weight.clone()

    for candidate, score in zip(candidate_weights, neighbor_scores):
        if selected_weights:
            previous_cos_score = cos_score
        selected_scores.append(float(score))
        selected_weights.append(candidate)
        vec_sum = torch.sum(torch.stack(selected_weights), dim=0)
        cos_score = torch.nn.functional.cosine_similarity(
            initial_clone.unsqueeze(0), vec_sum.unsqueeze(0), dim=1, eps=1e-6
        ).item()
        if not previous_cos_score < cos_score:
            selected_scores.pop()
            selected_weights.pop()
            break

    if len(selected_weights) <= 1:
        return initial_weight, 0

    normalized_weights = [
        initial_clone / torch.norm(initial_clone),
        *[value / torch.norm(value) for value in selected_weights if torch.norm(value) > 0],
    ]

    if method == "maximum_absolute":
        new_weight = maximum_absolute_values(torch.stack(normalized_weights))
        new_weight = new_weight * pre_mag / torch.norm(new_weight)
        return new_weight, len(selected_weights)

    if method == "add_minimum_absolute":
        minimum_weight = maximum_absolute_values(torch.stack(normalized_weights), reversed=True)
        new_weight = initial_clone + minimum_weight * float(intensity)
        new_weight = new_weight * pre_mag / torch.norm(new_weight)
        return new_weight, len(selected_weights)

    weighted_neighbors = torch.sum(
        torch.stack([value * (selected_scores[index] ** 2) for index, value in enumerate(selected_weights)]),
        dim=0,
    )
    final_score = torch.nn.functional.cosine_similarity(
        initial_weight.unsqueeze(0), weighted_neighbors.unsqueeze(0), dim=1, eps=1e-6
    ).item() * float(intensity)
    new_weight = (
        initial_weight + weighted_neighbors * final_score
        if method == "backward"
        else initial_weight - weighted_neighbors * final_score
    )
    new_norm = torch.norm(new_weight)
    if new_norm == 0:
        return initial_weight, 0
    return new_weight * pre_mag / new_norm, len(selected_weights)


class EmbeddingSculptSession:
    def __init__(self, weight, token_ids=(), top_k=64, preferred_device=None):
        self.weight = weight
        self.top_k = int(top_k)
        self.preferred_device = preferred_device
        self.neighbors, self.search_stats = chunked_top_neighbors(
            weight, token_ids, self.top_k, preferred_device=preferred_device
        )
        self._original_cache = {}
        self._sculpt_cache = {}
        self._mean_magnitude = None
        self._mean_stats = None

    def original(self, token_id):
        token_id = int(token_id)
        if token_id not in self._original_cache:
            self._original_cache[token_id] = _embedding_rows(
                self.weight, [token_id], torch.device("cpu")
            )[0]
        return self._original_cache[token_id]

    def sculpt(self, token_id, method, intensity):
        key = (int(token_id), str(method), float(intensity))
        if key not in self._sculpt_cache:
            neighbor_ids, neighbor_scores = self.neighbors.get(int(token_id), ([], []))
            self._sculpt_cache[key] = refine_from_neighbors(
                self.weight,
                token_id,
                neighbor_ids,
                neighbor_scores,
                method,
                intensity,
            )
        return self._sculpt_cache[key]

    def mean_magnitude(self):
        if self._mean_magnitude is None:
            self._mean_magnitude, self._mean_stats = streaming_mean_magnitude(
                self.weight, preferred_device=self.preferred_device
            )
        return self._mean_magnitude

    @property
    def cache_entries(self):
        return len(self._sculpt_cache)

    def report(self):
        stats = self.search_stats
        label = stats["device"]
        if stats.get("cpu_fallback"):
            label += " (OOM -> CPU fallback)"
        report = (
            f"search device: {label}; unique tokens: {stats['unique_tokens']}; "
            f"chunk rows: {stats['chunk_rows']}; query batches: {stats['query_batches']}"
        )
        if "elapsed_seconds" in stats:
            report += f"; search time: {stats['elapsed_seconds']:.2f}s"
        if self._mean_stats is not None:
            mean_label = self._mean_stats["device"]
            if self._mean_stats.get("cpu_fallback"):
                mean_label += " (OOM -> CPU fallback)"
            report += (
                f"; mean device: {mean_label}; mean chunk rows: {self._mean_stats['chunk_rows']}"
            )
        return report
