import numbers

import torch

from .embedding_sculpt_core import EmbeddingSculptSession
from .t5_equal_length_balancer import SUPPORTED_TEXT_STREAM_KEYS


SCULPT_METHODS = ("forward", "backward", "maximum_absolute", "add_minimum_absolute")
NORMALIZATION_MODES = ("none", "mean", "mean * attention")
QWEN3VL_IM_START = 151644
QWEN3VL_IM_END = 151645
QWEN3VL_USER = 872
QWEN3VL_NEWLINE = 198


class OnekoT5SculptEqualLengthBalancer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tokenizer": ("CLIP", {"tooltip": "The loaded T5/Qwen tokenizer to use."}),
                "target": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Minimum shared token length for positive and negative prompts.",
                    },
                ),
                "positive": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "negative": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "positive_intensity": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
                "positive_method": (SCULPT_METHODS, {"default": "forward"}),
                "positive_normalization": (NORMALIZATION_MODES, {"default": "none"}),
                "negative_intensity": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
                "negative_method": (SCULPT_METHODS, {"default": "forward"}),
                "negative_normalization": (NORMALIZATION_MODES, {"default": "mean"}),
                "top_k": (
                    "INT",
                    {
                        "default": 64,
                        "min": 1,
                        "max": 512,
                        "step": 1,
                        "tooltip": "Number of nearest token vectors used for sculpting.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "INT", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "positive",
        "negative",
        "positive_raw_tokens",
        "negative_raw_tokens",
        "effective_target",
        "positive_sculpted_tokens",
        "negative_sculpted_tokens",
        "report",
    )
    FUNCTION = "balance"
    CATEGORY = "Oneko/03 Conditioning/Encode"
    DESCRIPTION = (
        "Encodes positive and negative prompts with equal T5/Qwen token length, "
        "optionally sculpting eligible text token embeddings before scheduled encoding."
    )

    def _set_text_stream_options(self, clip, min_length=None, min_padding=None):
        for key in SUPPORTED_TEXT_STREAM_KEYS:
            if min_length is not None:
                clip.set_tokenizer_option(f"{key}_min_length", min_length)
            if min_padding is not None:
                clip.set_tokenizer_option(f"{key}_min_padding", min_padding)

    def _detect_text_stream_key(self, *token_sets):
        available = set()
        for tokens in token_sets:
            if isinstance(tokens, dict):
                available.update(tokens.keys())

        for key in SUPPORTED_TEXT_STREAM_KEYS:
            if key in available:
                return key
        return None

    def _detect_sculptable_text_stream_key(self, clip, *token_sets):
        available = set()
        for tokens in token_sets:
            if isinstance(tokens, dict):
                available.update(tokens.keys())

        candidates = [key for key in SUPPORTED_TEXT_STREAM_KEYS if key in available]
        for key in candidates:
            submodel = getattr(clip.cond_stage_model, key, None)
            transformer = getattr(submodel, "transformer", None)
            if transformer is not None and hasattr(transformer, "get_input_embeddings"):
                return key
        return candidates[0] if candidates else None

    def _available_token_keys(self, *token_sets):
        available = set()
        for tokens in token_sets:
            if isinstance(tokens, dict):
                available.update(tokens.keys())
        if not available:
            return "none"
        return ", ".join(sorted(available))

    def _token_count(self, tokens, key):
        try:
            batches = tokens[key]
            if len(batches) == 0:
                return 0
            return len(batches[0])
        except (KeyError, TypeError):
            raise RuntimeError(f"ERROR: Token stream '{key}' was not found in tokenizer output.")

    def _get_text_submodel(self, clip, stream_key):
        submodel = getattr(clip.cond_stage_model, stream_key, None)
        if submodel is None:
            raise RuntimeError(f"ERROR: Text encoder submodel '{stream_key}' was not found on cond_stage_model.")
        if not hasattr(submodel, "transformer"):
            raise RuntimeError(f"ERROR: Text encoder submodel '{stream_key}' has no transformer.")
        if not hasattr(submodel.transformer, "get_input_embeddings"):
            raise RuntimeError(
                f"ERROR: Text encoder submodel '{stream_key}' has no input embeddings accessor."
            )
        return submodel

    def _special_token_ids(self, submodel):
        special_tokens = getattr(submodel, "special_tokens", {}) or {}
        return {
            token_id
            for token_id in (
                special_tokens.get("start"),
                special_tokens.get("end"),
                special_tokens.get("pad"),
            )
            if token_id is not None
        }

    def _embedding_weights(self, submodel):
        return submodel.transformer.get_input_embeddings().weight

    def _refine_token_weight(self, token_id, all_weights, method, intensity, top_k):
        session = EmbeddingSculptSession(all_weights, [token_id], top_k)
        return session.sculpt(token_id, method, intensity)

    def _qwen3vl_user_content_range(self, batch):
        token_ids = [
            int(entry[0]) if len(entry) >= 1 and isinstance(entry[0], numbers.Integral) else None
            for entry in batch
        ]
        marker = (QWEN3VL_IM_START, QWEN3VL_USER, QWEN3VL_NEWLINE)
        for index in range(len(token_ids) - len(marker) + 1):
            if tuple(token_ids[index : index + len(marker)]) != marker:
                continue
            content_start = index + len(marker)
            try:
                content_end = token_ids.index(QWEN3VL_IM_END, content_start)
            except ValueError:
                return None
            return content_start, content_end
        return None

    def _eligible_entries(self, tokens, stream_key, special_ids):
        coords = []
        for batch_index, batch in enumerate(tokens[stream_key]):
            content_range = None
            if stream_key.startswith("qwen3vl_"):
                content_range = self._qwen3vl_user_content_range(batch)
            for token_index, token_weight in enumerate(batch):
                if content_range is not None and not content_range[0] <= token_index < content_range[1]:
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

    def _apply_sculpting(
        self,
        tokens,
        stream_key,
        submodel,
        intensity,
        method,
        normalization,
        top_k,
        session=None,
        coords=None,
    ):
        if float(intensity) <= 0 and normalization == "none":
            return 0, 0

        if stream_key not in tokens:
            raise RuntimeError(f"ERROR: Token stream '{stream_key}' was not found for sculpting.")

        special_ids = self._special_token_ids(submodel)
        coords = coords if coords is not None else self._eligible_entries(tokens, stream_key, special_ids)
        if len(coords) == 0:
            return 0, 0

        if session is None:
            search_ids = [token_id for _batch, _index, token_id in coords] if float(intensity) > 0 else []
            session = EmbeddingSculptSession(self._embedding_weights(submodel), search_ids, top_k)
        sculpted_count = 0
        neighbor_count = 0
        mean_mag = 0.0
        mean_coords = []

        for batch_index, token_index, token_id in coords:
            token_weight = tokens[stream_key][batch_index][token_index]
            attn_weight = float(token_weight[1])
            if float(intensity) > 0:
                new_vector, found = session.sculpt(token_id, method, float(intensity))
                if found > 0:
                    sculpted_count += 1
                    neighbor_count += found
            else:
                new_vector = session.original(token_id)

            new_vector = new_vector.clone()

            if normalization in ("mean", "mean * attention"):
                mean_mag += torch.norm(new_vector).item()
                mean_coords.append((batch_index, token_index, attn_weight))

            tokens[stream_key][batch_index][token_index] = (new_vector, attn_weight)

        if normalization in ("mean", "mean * attention") and len(mean_coords) > 0:
            mean_mag /= len(mean_coords)
            for batch_index, token_index, attn_weight in mean_coords:
                token_vector, current_attn = tokens[stream_key][batch_index][token_index]
                norm = torch.norm(token_vector)
                if norm == 0:
                    continue
                scale = mean_mag * (attn_weight if normalization == "mean * attention" else 1.0)
                tokens[stream_key][batch_index][token_index] = (token_vector / norm * scale, current_attn)

        return sculpted_count, neighbor_count

    def balance(
        self,
        tokenizer,
        target,
        positive,
        negative,
        positive_intensity,
        positive_method,
        positive_normalization,
        negative_intensity,
        negative_method,
        negative_normalization,
        top_k,
    ):
        if tokenizer is None:
            raise RuntimeError("ERROR: A valid tokenizer is required.")

        measure = tokenizer.clone()
        self._set_text_stream_options(measure, min_length=0, min_padding=0)
        raw_positive_tokens = measure.tokenize(positive)
        raw_negative_tokens = measure.tokenize(negative)

        stream_key = self._detect_sculptable_text_stream_key(measure, raw_positive_tokens, raw_negative_tokens)
        if stream_key is None:
            expected = ", ".join(SUPPORTED_TEXT_STREAM_KEYS)
            available = self._available_token_keys(raw_positive_tokens, raw_negative_tokens)
            raise RuntimeError(
                "ERROR: No supported text token stream found. "
                f"Expected one of: {expected}. Available token streams: {available}."
            )

        positive_raw_count = self._token_count(raw_positive_tokens, stream_key)
        negative_raw_count = self._token_count(raw_negative_tokens, stream_key)
        requested_target = max(0, int(target))
        effective_target = max(requested_target, positive_raw_count, negative_raw_count)

        encoder = tokenizer.clone()
        self._set_text_stream_options(encoder, min_padding=0)
        encoder.set_tokenizer_option(f"{stream_key}_min_length", effective_target)
        submodel = self._get_text_submodel(encoder, stream_key)

        positive_tokens = encoder.tokenize(positive)
        negative_tokens = encoder.tokenize(negative)

        special_ids = self._special_token_ids(submodel)
        positive_coords = self._eligible_entries(positive_tokens, stream_key, special_ids)
        negative_coords = self._eligible_entries(negative_tokens, stream_key, special_ids)
        search_ids = []
        if float(positive_intensity) > 0:
            search_ids.extend(token_id for _batch, _index, token_id in positive_coords)
        if float(negative_intensity) > 0:
            search_ids.extend(token_id for _batch, _index, token_id in negative_coords)
        session = EmbeddingSculptSession(
            self._embedding_weights(submodel),
            search_ids,
            top_k,
        )

        positive_sculpted, positive_neighbors = self._apply_sculpting(
            positive_tokens,
            stream_key,
            submodel,
            positive_intensity,
            positive_method,
            positive_normalization,
            top_k,
            session=session,
            coords=positive_coords,
        )
        negative_sculpted, negative_neighbors = self._apply_sculpting(
            negative_tokens,
            stream_key,
            submodel,
            negative_intensity,
            negative_method,
            negative_normalization,
            top_k,
            session=session,
            coords=negative_coords,
        )

        cond_positive = encoder.encode_from_tokens_scheduled(positive_tokens)
        cond_negative = encoder.encode_from_tokens_scheduled(negative_tokens)

        report = (
            f"text stream: {stream_key}; "
            f"positive raw: {positive_raw_count}; "
            f"negative raw: {negative_raw_count}; "
            f"requested target: {requested_target}; "
            f"effective target: {effective_target}; "
            f"positive sculpted: {positive_sculpted} tokens / {positive_neighbors} neighbors; "
            f"positive method: {positive_method}; positive intensity: {float(positive_intensity):.2f}; "
            f"positive normalization: {positive_normalization}; "
            f"negative sculpted: {negative_sculpted} tokens / {negative_neighbors} neighbors; "
            f"negative method: {negative_method}; negative intensity: {float(negative_intensity):.2f}; "
            f"negative normalization: {negative_normalization}; "
            f"top_k: {int(top_k)}; {session.report()}"
        )

        return (
            cond_positive,
            cond_negative,
            positive_raw_count,
            negative_raw_count,
            effective_target,
            positive_sculpted,
            negative_sculpted,
            report,
        )


NODE_CLASS_MAPPINGS = {
    "OnekoT5SculptEqualLengthBalancer": OnekoT5SculptEqualLengthBalancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnekoT5SculptEqualLengthBalancer": "T5/Qwen Sculpt Equal-Length Prompt Balancer (Oneko)"
}
