from .clip_sculpt_core import NORMALIZATION_MODES, SCULPT_METHODS, sculpt_clip_tokens


class OnekoCLIPSculptTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
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
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "report")
    FUNCTION = "encode"
    CATEGORY = "Oneko/03 Conditioning/Encode"
    DESCRIPTION = (
        "SD1/SDXL CLIP text encoder with local Vector-Sculptor-style token embedding edits "
        "and scheduled CLIP encoding support."
    )

    def encode(self, clip, text, sculptor_intensity, sculptor_method, token_normalization, top_k):
        if clip is None:
            raise RuntimeError("ERROR: A valid CLIP input is required.")

        tokens, stats = sculpt_clip_tokens(
            clip,
            text,
            sculptor_intensity,
            sculptor_method,
            token_normalization,
            top_k,
        )
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if float(sculptor_intensity) == 0.0 and token_normalization == "none":
            report = "disabled"
        else:
            report = (
                f"method: {sculptor_method}; intensity: {float(sculptor_intensity):.2f}; "
                f"normalization: {token_normalization}; top_k: {int(top_k)}; "
                f"eligible: {stats['eligible']}; sculpted: {stats['sculpted']}; "
                f"neighbors: {stats['neighbors']}; cache entries: {stats['cache_entries']}; "
                f"streams: {stats['streams']}; {stats['search']}"
            )
        return (conditioning, report)


NODE_CLASS_MAPPINGS = {
    "OnekoCLIPSculptTextEncode": OnekoCLIPSculptTextEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnekoCLIPSculptTextEncode": "CLIP Sculpt Text Encode (Oneko)"
}
