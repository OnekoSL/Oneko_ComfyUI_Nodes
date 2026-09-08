class OnekoIncrementingIntString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xffffffffffffffff,
                        "control_after_generate": True,
                        "tooltip": "Integer value to output as a plain string. The frontend can increment it after queueing.",
                    },
                ),
                "min_value": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xffffffffffffffff,
                        "tooltip": "Lowest value in the output cycle.",
                    },
                ),
                "max_value": (
                    "INT",
                    {
                        "default": 9999,
                        "min": 0,
                        "max": 0xffffffffffffffff,
                        "tooltip": "Highest value in the output cycle. If it is lower than min_value, the node swaps both values.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("STRING", "INT", "raw_value")
    FUNCTION = "convert"
    CATEGORY = "Oneko/10 Utilities"
    DESCRIPTION = "Outputs an integer as a plain string, wrapped between min_value and max_value, with control-after-generate support."

    def convert(self, value, min_value, max_value):
        low = min(min_value, max_value)
        high = max(min_value, max_value)
        cycle_size = high - low + 1
        wrapped = low + ((value - low) % cycle_size)
        return (f"{wrapped}", wrapped, value)


NODE_CLASS_MAPPINGS = {
    "OnekoIncrementingIntString": OnekoIncrementingIntString,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnekoIncrementingIntString": "Incrementing Int to String (Oneko)"
}
