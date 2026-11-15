"""Token provenance for protected post-encoding conditioning adjustments.

Only native MOSS plain-text token streams are annotated. The metadata is a
description of token positions, not a model guess from embedding dimensions.
"""

import hashlib
import numbers

import torch

from .clip_sculpt_core import eligible_entries


CONDITIONING_CONTEXT_KEY = "oneko_conditioning_context"


def conditioning_fingerprint(tensor):
    """Bind token-position provenance to exact conditioning values, not storage."""
    if not torch.is_tensor(tensor):
        raise TypeError("Conditioning fingerprint requires a tensor.")
    raw = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(f"{raw.dtype}:{tuple(raw.shape)}:".encode("ascii"))
    # Byte views support BF16 and avoid a second host copy through tobytes().
    digest.update(memoryview(raw.view(torch.uint8).numpy()))
    return digest.hexdigest()


def capture_conditioning_context(clip, tokens):
    # Local import avoids a cycle with the balancer's encode integration.
    from .t5_equal_length_balancer import requires_native_text_lengths

    if not requires_native_text_lengths(clip):
        return None

    context = {
        "version": 1,
        "model_profile": "moss_image",
        "token_mask": None,
        "token_count": 0,
        "warning": "",
    }
    stream_key = "qwen35_9b"
    batches = tokens.get(stream_key) if isinstance(tokens, dict) else None
    if not isinstance(batches, (list, tuple)) or not batches:
        context["warning"] = "MOSS token provenance unavailable: missing plain-text token sections."
        return context
    if any(not isinstance(batch, (list, tuple)) for batch in batches):
        context["warning"] = "MOSS token provenance unavailable: invalid token sections."
        return context

    context["token_count"] = sum(len(batch) for batch in batches)
    if any(
        not isinstance(entry, (list, tuple))
        or len(entry) < 2
        or not isinstance(entry[0], numbers.Integral)
        for batch in batches for entry in batch
    ):
        # Embeddings and multimodal entries can expand to multiple output rows.
        # Capture before Sculpt replaces ordinary token IDs with 1-row vectors.
        context["warning"] = (
            "MOSS token provenance unavailable: embedded or multimodal tokens "
            "cannot be mapped safely to conditioning rows."
        )
        return context

    stream_tokenizer = getattr(getattr(clip, "tokenizer", None), stream_key, None)
    tokenizer = getattr(stream_tokenizer, "tokenizer", None)
    if tokenizer is None:
        context["warning"] = "MOSS token provenance unavailable: missing tokenizer metadata."
        return context
    try:
        coords = eligible_entries(
            tokens, stream_key, set(tokenizer.all_special_ids), clip=clip
        )
    except (AttributeError, KeyError, TypeError, ValueError, IndexError):
        context["warning"] = "MOSS token provenance unavailable: unreadable tokenizer metadata."
        return context

    masks = [[False] * len(batch) for batch in batches]
    for section, index, _token_id in coords:
        masks[section][index] = True
    context["token_mask"] = tuple(value for mask in masks for value in mask)
    if not coords:
        context["warning"] = (
            "MOSS conditioning has no eligible user text; empty or "
            "unrecognized/incomplete chat remains unchanged."
        )
    return context


def attach_conditioning_context(conditioning, context):
    if context is None:
        return conditioning

    output = []
    for tensor, metadata in conditioning:
        attached = dict(context)
        token_mask = attached.get("token_mask")
        if token_mask is not None and (
            not torch.is_tensor(tensor)
            or tensor.ndim != 3
            or tensor.shape[1] != attached["token_count"]
            or len(token_mask) != attached["token_count"]
        ):
            attached["token_mask"] = None
            attached["warning"] = (
                "MOSS token provenance unavailable: encoded conditioning length "
                "does not match the captured plain-text tokens."
            )
        attached["source_fingerprint"] = (
            conditioning_fingerprint(tensor) if attached.get("token_mask") is not None else None
        )
        copied_metadata = dict(metadata)
        copied_metadata[CONDITIONING_CONTEXT_KEY] = attached
        output.append([tensor, copied_metadata])
    return output
