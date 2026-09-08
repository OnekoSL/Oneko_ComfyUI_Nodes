SUPPORTED_TEXT_STREAM_KEYS = (
    "pile_t5xl",
    "t5xxl",
    "umt5xxl",
    "mt5xl",
    "t5base",
    "qwen3_4b",
    "qwen3_8b",
    "qwen25_7b",
    "qwen25_3b",
    "qwen3_2b",
    "qwen3_06b",
    "qwen3vl_4b",
    "qwen3vl_8b",
)

# Backwards-compatible name for older imports.
SUPPORTED_T5_KEYS = SUPPORTED_TEXT_STREAM_KEYS


class OnekoT5EqualLengthBalancer:
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
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "positive",
        "negative",
        "positive_raw_tokens",
        "negative_raw_tokens",
        "effective_target",
        "report",
    )
    FUNCTION = "balance"
    CATEGORY = "Oneko/03 Conditioning/Encode"
    DESCRIPTION = (
        "Encodes positive and negative prompts with a supported T5/Qwen tokenizer, "
        "padding both to the same effective token length and returning token diagnostics."
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

    def _detect_conditioning_text_stream_key(self, clip, *token_sets):
        available = set()
        for tokens in token_sets:
            if isinstance(tokens, dict):
                available.update(tokens.keys())

        candidates = [key for key in SUPPORTED_TEXT_STREAM_KEYS if key in available]
        for key in candidates:
            if getattr(clip.cond_stage_model, key, None) is not None:
                return key
        return candidates[0] if candidates else None

    def _token_count(self, tokens, key):
        try:
            batches = tokens[key]
            if len(batches) == 0:
                return 0
            return len(batches[0])
        except (KeyError, TypeError):
            raise RuntimeError(f"ERROR: Token stream '{key}' was not found in tokenizer output.")

    def _available_token_keys(self, *token_sets):
        available = set()
        for tokens in token_sets:
            if isinstance(tokens, dict):
                available.update(tokens.keys())
        if not available:
            return "none"
        return ", ".join(sorted(available))

    def balance(self, tokenizer, target, positive, negative):
        if tokenizer is None:
            raise RuntimeError("ERROR: A valid tokenizer is required.")

        measure = tokenizer.clone()
        self._set_text_stream_options(measure, min_length=0, min_padding=0)
        raw_positive_tokens = measure.tokenize(positive)
        raw_negative_tokens = measure.tokenize(negative)

        stream_key = self._detect_conditioning_text_stream_key(measure, raw_positive_tokens, raw_negative_tokens)
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

        positive_tokens = encoder.tokenize(positive)
        negative_tokens = encoder.tokenize(negative)
        cond_positive = encoder.encode_from_tokens_scheduled(positive_tokens)
        cond_negative = encoder.encode_from_tokens_scheduled(negative_tokens)

        report = (
            f"text stream: {stream_key}; "
            f"positive raw: {positive_raw_count}; "
            f"negative raw: {negative_raw_count}; "
            f"requested target: {requested_target}; "
            f"effective target: {effective_target}"
        )

        return (
            cond_positive,
            cond_negative,
            positive_raw_count,
            negative_raw_count,
            effective_target,
            report,
        )


NODE_CLASS_MAPPINGS = {
    "OnekoT5EqualLengthBalancer": OnekoT5EqualLengthBalancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnekoT5EqualLengthBalancer": "T5/Qwen Equal-Length Prompt Balancer (Oneko)"
}
