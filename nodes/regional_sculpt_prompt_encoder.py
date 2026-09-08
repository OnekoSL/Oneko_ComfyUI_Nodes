from .clip_sculpt_core import (
    NORMALIZATION_MODES,
    SCULPT_METHODS,
    sculpt_clip_token_sets,
)
from .regional_prompt_encoder import (
    _clean_text,
    _combine_hiresfix_text,
    _combine_text,
    _encode_text,
)


class OnekoRegionalSculptPromptEncoder:
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
                "sculptor_intensity": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
                "sculptor_method": (SCULPT_METHODS, {"default": "forward"}),
                "token_normalization": (NORMALIZATION_MODES, {"default": "mean"}),
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
        "report",
    )
    FUNCTION = "encode"
    CATEGORY = "Oneko/04 Regional/Encode"
    DESCRIPTION = (
        "Combines base and regional prompts, then encodes SDXL/CLIP regional prompts "
        "with low-memory Vector-Sculptor-style token embedding edits."
    )

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
        sculptor_intensity,
        sculptor_method,
        token_normalization,
        top_k,
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

        sculpt_texts = [combined_1, combined_2]
        if combined_3:
            sculpt_texts.append(combined_3)
        hires_index = len(sculpt_texts)
        sculpt_texts.append(hiresfix_text)
        token_sets = [clip.tokenize(text) for text in sculpt_texts]
        token_sets, stats = sculpt_clip_token_sets(
            clip,
            token_sets,
            sculptor_intensity,
            sculptor_method,
            token_normalization,
            top_k,
            strict=True,
            scale_clip_g=False,
        )
        conditionings = [clip.encode_from_tokens_scheduled(tokens) for tokens in token_sets]

        base_conditioning = _encode_text(clip, base_text)
        conditioning_1 = conditionings[0]
        conditioning_2 = conditionings[1]
        if combined_3:
            conditioning_3 = conditionings[2]
            stats_3 = stats[2]
        else:
            conditioning_3 = _encode_text(clip, base_text)
            stats_3 = {
                "eligible": 0,
                "sculpted": 0,
                "neighbors": 0,
                "streams": "base fallback",
                "search": "disabled",
            }
        hiresfix_conditioning = conditionings[hires_index]

        total_eligible = sum(item["eligible"] for item in stats)
        total_sculpted = sum(item["sculpted"] for item in stats)
        total_neighbors = sum(item["neighbors"] for item in stats)
        cache_entries = max((item["cache_entries"] for item in stats), default=0)
        search_reports = " | ".join(
            dict.fromkeys(item["search"] for item in stats if item["search"] != "disabled")
        ) or "disabled"
        report = (
            f"method: {sculptor_method}; intensity: {float(sculptor_intensity):.2f}; "
            f"normalization: {token_normalization}; top_k: {int(top_k)}; "
            f"eligible: {total_eligible}; sculpted: {total_sculpted}; "
            f"neighbors: {total_neighbors}; cache entries: {cache_entries}; "
            f"region_1 streams: {stats[0]['streams']}; region_2 streams: {stats[1]['streams']}; "
            f"region_3 streams: {stats_3['streams']}; hires streams: {stats[hires_index]['streams']}; "
            f"{search_reports}"
        )

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
            report,
        )


NODE_CLASS_MAPPINGS = {
    "OnekoRegionalSculptPromptEncoder": OnekoRegionalSculptPromptEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnekoRegionalSculptPromptEncoder": "Regional Sculpt Prompt Encoder (Oneko)"
}
