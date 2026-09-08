class OnekoRegionalPromptEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "base_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "region_1": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "region_2": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "region_3": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "region_count": ("INT", {"default": 2, "min": 2, "max": 3, "step": 1}),
                "separator": ("STRING", {"default": ", "}),
                "hiresfix_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
        }

    RETURN_TYPES = (
        "CONDITIONING",
        "CONDITIONING",
        "CONDITIONING",
        "CONDITIONING",
        "CONDITIONING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "base_conditioning",
        "conditioning_1",
        "conditioning_2",
        "conditioning_3",
        "hiresfix_conditioning",
        "base_text",
        "text_1",
        "text_2",
        "text_3",
        "hiresfix_text",
    )
    FUNCTION = "encode"
    CATEGORY = "Oneko/04 Regional/Encode"
    DESCRIPTION = "Combines a base prompt with 2/3 regional prompts and encodes them with CLIP."

    def encode(
        self,
        clip,
        base_prompt,
        region_1,
        region_2,
        region_3,
        region_count,
        separator,
        hiresfix_prompt,
    ):
        if clip is None:
            raise RuntimeError("ERROR: A valid CLIP input is required.")

        region_count = 3 if region_count >= 3 else 2
        base_text = _clean_text(base_prompt)
        combined_1 = _combine_text(base_text, region_1, separator)
        combined_2 = _combine_text(base_text, region_2, separator)
        combined_3 = _combine_text(base_text, region_3, separator) if region_count == 3 else ""
        hiresfix_text = _combine_hiresfix_text(
            base_text,
            [region_1, region_2, region_3 if region_count == 3 else ""],
            hiresfix_prompt,
            separator,
        )

        base_conditioning = _encode_text(clip, base_text)
        conditioning_1 = _encode_text(clip, combined_1)
        conditioning_2 = _encode_text(clip, combined_2)
        conditioning_3 = _encode_text(clip, combined_3 if combined_3 else base_text)
        hiresfix_conditioning = _encode_text(clip, hiresfix_text)

        return (
            base_conditioning,
            conditioning_1,
            conditioning_2,
            conditioning_3,
            hiresfix_conditioning,
            base_text,
            combined_1,
            combined_2,
            combined_3,
            hiresfix_text,
        )


def _clean_text(text):
    return str(text or "").strip()


def _combine_text(base_text, region_text, separator):
    base_text = _clean_text(base_text)
    region_text = _clean_text(region_text)
    separator = str(separator if separator is not None else ", ")

    if base_text and region_text:
        return f"{base_text}{separator}{region_text}"
    return base_text or region_text


def _combine_hiresfix_text(base_text, region_texts, hiresfix_prompt, separator):
    separator = str(separator if separator is not None else ", ")
    parts = [base_text]
    parts.extend(_clean_text(text) for text in region_texts)
    parts.append(_clean_text(hiresfix_prompt))
    return separator.join(part for part in parts if part)


def _encode_text(clip, text):
    tokens = clip.tokenize(text)
    return clip.encode_from_tokens_scheduled(tokens)


NODE_CLASS_MAPPINGS = {
    "OnekoRegionalPromptEncoder": OnekoRegionalPromptEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnekoRegionalPromptEncoder": "Regional Prompt Encoder (Oneko)"
}
