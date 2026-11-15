import numbers

import torch

from .embedding_sculpt_core import EmbeddingSculptSession


SCULPT_METHODS = ("forward", "backward", "maximum_absolute", "add_minimum_absolute")
NORMALIZATION_MODES = (
    "none",
    "mean",
    "set at 1",
    "default * attention",
    "mean * attention",
    "set at attention",
    "mean of all tokens",
)


def get_clip_submodel(clip, stream_key):
    candidates = (f"clip_{stream_key}", stream_key)
    submodel = None
    matched_name = None
    for candidate in candidates:
        submodel = getattr(clip.cond_stage_model, candidate, None)
        if submodel is not None:
            matched_name = candidate
            break
    if submodel is None:
        raise RuntimeError(
            f"ERROR: CLIP stream '{stream_key}' is not supported by this node. "
            f"Expected one of: {', '.join('cond_stage_model.' + name for name in candidates)}."
        )
    if not hasattr(submodel, "transformer"):
        raise RuntimeError(f"ERROR: CLIP stream '{stream_key}' on cond_stage_model.{matched_name} has no transformer.")
    embedding = get_input_embedding(submodel, stream_key)
    if not hasattr(embedding, "weight"):
        raise RuntimeError(
            f"ERROR: CLIP stream '{stream_key}' has no token embedding table. "
            "This node is for SD1/SDXL CLIP-style encoders, not T5."
        )
    return submodel


def maybe_get_clip_submodel(clip, stream_key):
    try:
        return get_clip_submodel(clip, stream_key), None
    except RuntimeError as error:
        return None, str(error)


def get_input_embedding(submodel, stream_key):
    transformer = submodel.transformer
    if hasattr(transformer, "get_input_embeddings"):
        return transformer.get_input_embeddings()
    try:
        return transformer.text_model.embeddings.token_embedding
    except AttributeError:
        raise RuntimeError(
            f"ERROR: CLIP stream '{stream_key}' has no supported input embedding accessor."
        )


def special_token_ids(submodel):
    special_tokens = getattr(submodel, "special_tokens", None)
    if special_tokens:
        return {token_id for token_id in special_tokens.values() if token_id is not None}
    return {49406, 49407, 0}


def embedding_weights(submodel, stream_key):
    return get_input_embedding(submodel, stream_key).weight


def refine_token_weight(token_id, all_weights, method, intensity, top_k):
    session = EmbeddingSculptSession(all_weights, [token_id], top_k)
    return session.sculpt(token_id, method, intensity)


def _chat_user_ranges(batch, tokenizer):
    """Return only complete user turns; malformed chats are left untouched."""
    token_ids = [
        int(entry[0]) if entry and isinstance(entry[0], numbers.Integral) else None
        for entry in batch
    ]
    start = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if len(start) != 1 or len(end) != 1:
        return []

    def content_start(index, role):
        header = tokenizer.encode("<|im_start|>" + role, add_special_tokens=False)
        separator_index = index + len(header)
        if token_ids[index:separator_index] != header or separator_index >= len(token_ids):
            return None
        separator_id = token_ids[separator_index]
        if separator_id is None:
            return None
        separator = tokenizer.decode([separator_id])
        # Newline runs can be merged by BPE. Keep the whole separator untouched.
        if not separator.startswith("\n") or separator.strip():
            return None
        return separator_index + 1

    starts = [index for index, token_id in enumerate(token_ids) if token_id == start[0]]
    ranges = []
    for turn, index in enumerate(starts):
        next_start = starts[turn + 1] if turn + 1 < len(starts) else len(token_ids)
        try:
            turn_end = token_ids.index(end[0], index + 1, next_start)
        except ValueError:
            # Native templates deliberately end with an open assistant/think turn.
            if turn == len(starts) - 1 and content_start(index, "assistant") is not None:
                break
            return []
        user_start = content_start(index, "user")
        if user_start is not None and user_start <= turn_end:
            ranges.append((user_start, turn_end))
    return ranges


def eligible_entries(tokens, stream_key, special_ids, clip=None):
    chat_stream = stream_key.startswith(("qwen35_", "qwen3vl_"))
    tokenizer = None
    if chat_stream:
        stream_tokenizer = getattr(getattr(clip, "tokenizer", None), stream_key, None)
        tokenizer = getattr(stream_tokenizer, "tokenizer", None)
        if tokenizer is None:
            return []
        # Some structural tokens (including <think>) are added but not special.
        special_ids = set(special_ids) | set(tokenizer.get_added_vocab().values()) | set(tokenizer.all_special_ids)

    coords = []
    for batch_index, batch in enumerate(tokens[stream_key]):
        ranges = _chat_user_ranges(batch, tokenizer) if chat_stream else None
        for token_index, token_weight in enumerate(batch):
            if ranges is not None and not any(start <= token_index < end for start, end in ranges):
                continue
            if len(token_weight) < 2:
                continue
            token_id = token_weight[0]
            if not isinstance(token_id, numbers.Integral):
                continue
            if int(token_id) in special_ids:
                continue
            coords.append((batch_index, token_index, int(token_id)))
    return coords


def apply_mean_normalization(tokens, stream_key, coords, normalization):
    if normalization not in ("mean", "mean * attention"):
        return

    mean_mag = 0.0
    mean_coords = []
    for batch_index, token_index, _ in coords:
        token_vector, attn_weight = tokens[stream_key][batch_index][token_index]
        if not torch.is_tensor(token_vector):
            continue
        mean_mag += torch.norm(token_vector).item()
        mean_coords.append((batch_index, token_index, float(attn_weight)))

    if len(mean_coords) == 0:
        return

    mean_mag /= len(mean_coords)
    for batch_index, token_index, attn_weight in mean_coords:
        token_vector, current_attn = tokens[stream_key][batch_index][token_index]
        norm = torch.norm(token_vector)
        if norm == 0:
            continue
        scale = mean_mag * (attn_weight if normalization == "mean * attention" else 1.0)
        tokens[stream_key][batch_index][token_index] = (token_vector / norm * scale, current_attn)


def _empty_stats():
    return {
        "eligible": 0,
        "sculpted": 0,
        "neighbors": 0,
        "cache_entries": 0,
        "streams": [],
        "search": [],
    }


def _apply_session(
    tokens,
    stream_key,
    coords,
    session,
    intensity,
    method,
    normalization,
    mean_all,
    scale_clip_g,
):
    sculpted = 0
    neighbors = 0
    actual_intensity = (
        float(intensity) * 4.0 / 1.5
        if scale_clip_g and stream_key.lower() == "g"
        else float(intensity)
    )

    for batch_index, token_index, token_id in coords:
        token_weight = tokens[stream_key][batch_index][token_index]
        attn_weight = float(token_weight[1])
        if float(intensity) > 0:
            new_vector, found = session.sculpt(token_id, method, actual_intensity)
            if found > 0:
                sculpted += 1
                neighbors += found
        else:
            new_vector = session.original(token_id)

        new_vector = new_vector.clone()
        if normalization == "set at 1":
            norm = torch.norm(new_vector)
            if norm > 0:
                new_vector = new_vector / norm
        elif normalization == "default * attention":
            new_vector = new_vector * attn_weight
        elif normalization == "set at attention":
            norm = torch.norm(new_vector)
            if norm > 0:
                new_vector = new_vector / norm * attn_weight
        elif normalization == "mean of all tokens":
            norm = torch.norm(new_vector)
            if norm > 0:
                new_vector = new_vector / norm * mean_all

        tokens[stream_key][batch_index][token_index] = (new_vector, attn_weight)

    apply_mean_normalization(tokens, stream_key, coords, normalization)
    return sculpted, neighbors


def sculpt_clip_token_sets(
    clip,
    token_sets,
    intensity,
    method,
    normalization,
    top_k,
    strict=False,
    scale_clip_g=True,
):
    token_sets = list(token_sets)
    stats = [_empty_stats() for _ in token_sets]
    if float(intensity) <= 0 and normalization == "none":
        if strict:
            for tokens in token_sets:
                for stream_key in tokens:
                    get_clip_submodel(clip, stream_key)
        for item in stats:
            item["streams"] = "disabled"
            item["search"] = "disabled"
        return token_sets, stats

    stream_keys = []
    for tokens in token_sets:
        for stream_key in tokens:
            if stream_key not in stream_keys:
                stream_keys.append(stream_key)

    for stream_key in stream_keys:
        if strict:
            submodel = get_clip_submodel(clip, stream_key)
        else:
            submodel, _skip_reason = maybe_get_clip_submodel(clip, stream_key)
            if submodel is None:
                for index, tokens in enumerate(token_sets):
                    if stream_key in tokens:
                        stats[index]["streams"].append(f"{stream_key}: skipped unsupported")
                continue

        chat_policy = " (complete user turns only; unrecognized/incomplete chat skipped)" if stream_key.startswith(("qwen35_", "qwen3vl_")) else ""
        special_ids = special_token_ids(submodel)
        coords_by_set = [
            eligible_entries(tokens, stream_key, special_ids, clip=clip) if stream_key in tokens else []
            for tokens in token_sets
        ]
        unique_ids = list(
            dict.fromkeys(token_id for coords in coords_by_set for _batch, _index, token_id in coords)
        )
        search_ids = unique_ids if float(intensity) > 0 else []
        weight = embedding_weights(submodel, stream_key)
        session = EmbeddingSculptSession(weight, search_ids, top_k)
        mean_all = session.mean_magnitude() if normalization == "mean of all tokens" and unique_ids else None

        for index, (tokens, coords) in enumerate(zip(token_sets, coords_by_set)):
            if stream_key not in tokens:
                continue
            stats[index]["eligible"] += len(coords)
            if not coords:
                stats[index]["streams"].append(f"{stream_key}: 0 eligible" + chat_policy)
                continue
            sculpted, neighbors = _apply_session(
                tokens,
                stream_key,
                coords,
                session,
                intensity,
                method,
                normalization,
                mean_all,
                scale_clip_g,
            )
            stats[index]["sculpted"] += sculpted
            stats[index]["neighbors"] += neighbors
            stats[index]["streams"].append(f"{stream_key}: {len(coords)} eligible" + chat_policy)
            stats[index]["search"].append(session.report())

        cache_entries = session.cache_entries
        for item in stats:
            item["cache_entries"] += cache_entries

    for item in stats:
        item["streams"] = ", ".join(item["streams"]) or "none"
        item["search"] = " | ".join(dict.fromkeys(item["search"])) or "disabled"
    return token_sets, stats


def sculpt_clip_tokens(clip, text, intensity, method, normalization, top_k):
    token_sets, stats = sculpt_clip_token_sets(
        clip,
        [clip.tokenize(text)],
        intensity,
        method,
        normalization,
        top_k,
        strict=False,
    )
    return token_sets[0], stats[0]
