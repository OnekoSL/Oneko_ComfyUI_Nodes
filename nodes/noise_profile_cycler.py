from .noise_sampler_core import ILLUSTRIOUS_MODES, NOISE_TYPES, PONY_V7_PROFILES, UNIVERSAL_NOISE_PROFILES


LEGACY_BASIC_NOISE_TYPES = ["gaussian", "uniform", "laplacian", "pink", "brown", "blue", "violet", "pyramid", "perlin"]
EXPANDED_NOISE_TYPES = [profile for profile in NOISE_TYPES if profile not in LEGACY_BASIC_NOISE_TYPES]
ILLUSTRIOUS_PROFILES = [f"illustrious_{mode}" for mode in ILLUSTRIOUS_MODES]
PONY_V7_UNIVERSAL_PROFILES = [f"pony_v7_{profile}" for profile in PONY_V7_PROFILES]
COMPOSITE_PROFILES = ILLUSTRIOUS_PROFILES + PONY_V7_UNIVERSAL_PROFILES
PROFILE_SETS = {
    "all": list(UNIVERSAL_NOISE_PROFILES),
    "basic": list(NOISE_TYPES),
    "legacy_basic": LEGACY_BASIC_NOISE_TYPES,
    "expanded": EXPANDED_NOISE_TYPES,
    "composite": COMPOSITE_PROFILES,
    "illustrious": ILLUSTRIOUS_PROFILES,
    "pony_v7": PONY_V7_UNIVERSAL_PROFILES,
}


class OnekoNoiseProfileCycler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "profile_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 10000,
                        "control_after_generate": True,
                        "tooltip": "Incrementable test counter. The selected profile wraps inside start_index..end_index.",
                    },
                ),
                "profile_set": (
                    list(PROFILE_SETS.keys()),
                    {
                        "default": "all",
                        "tooltip": "Profile group to test. all includes basic and composite Universal Noise Sampler profiles.",
                    },
                ),
                "start_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 10000,
                        "tooltip": "First selectable index inside the chosen profile set.",
                    },
                ),
                "end_index": (
                    "INT",
                    {
                        "default": 10000,
                        "min": 0,
                        "max": 10000,
                        "tooltip": "Last selectable index inside the chosen profile set. Values beyond the set are clamped.",
                    },
                ),
            },
        }

    RETURN_TYPES = (UNIVERSAL_NOISE_PROFILES, "STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("noise_profile", "profile_name", "profile_index", "raw_index", "profile_count")
    FUNCTION = "select"
    CATEGORY = "Oneko/05 Sampling"
    DESCRIPTION = "Cycles through Universal Noise Sampler profiles with an incrementable index for systematic tests."

    def select(self, profile_index, profile_set="all", start_index=0, end_index=10000):
        profiles = PROFILE_SETS.get(profile_set, PROFILE_SETS["all"])
        count = len(profiles)
        low = max(0, min(int(start_index), count - 1))
        high = max(0, min(int(end_index), count - 1))
        if high < low:
            low, high = high, low

        cycle_size = high - low + 1
        wrapped_index = low + (int(profile_index) % cycle_size)
        profile_name = profiles[wrapped_index]
        return (profile_name, profile_name, wrapped_index, int(profile_index), count)


NODE_CLASS_MAPPINGS = {
    "OnekoNoiseProfileCycler": OnekoNoiseProfileCycler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnekoNoiseProfileCycler": "Noise Profile Cycler (Oneko)"
}
