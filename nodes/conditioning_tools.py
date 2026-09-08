from copy import deepcopy

import torch
import torch.nn.functional as F


PRESETS = [
    "neutral_report_only",
    "literal_detail",
    "soft_balance",
    "contrast_pop",
    "negative_tamer",
]

MODEL_PROFILES = [
    "auto",
    "t5",
    "sdxl_clip",
    "sd15_clip",
]

PRESET_PARAMS = {
    "literal_detail": {"center": 0.08, "contrast": 0.12, "smooth": 0.0, "spike": 0.0},
    "soft_balance": {"center": 0.12, "contrast": -0.15, "smooth": 0.35, "spike": 0.0},
    "contrast_pop": {"center": 0.05, "contrast": 0.25, "smooth": 0.0, "spike": 0.0},
    "negative_tamer": {"center": 0.10, "contrast": -0.25, "smooth": 0.0, "spike": 0.50},
}

PROFILE_SCALES = {
    "t5": {"center": 1.0, "contrast": 1.0, "smooth": 1.0, "spike": 1.0},
    "sdxl_clip": {"center": 0.55, "contrast": 0.55, "smooth": 0.70, "spike": 0.65},
    "sd15_clip": {"center": 0.45, "contrast": 0.45, "smooth": 0.65, "spike": 0.55},
    "generic": {"center": 0.40, "contrast": 0.40, "smooth": 0.55, "spike": 0.50},
}


def _conditioning_report(conditioning, title, profile_request="auto", rms_pairs=None):
    lines = [title, f"entries: {len(conditioning) if conditioning else 0}"]
    if not conditioning:
        return "\n".join(lines)

    rms_pairs = rms_pairs or {}
    for index, entry in enumerate(conditioning):
        if not entry or len(entry) < 2:
            lines.append(f"[{index}] invalid conditioning entry")
            continue

        tensor, meta = entry
        meta_keys = sorted(meta.keys()) if isinstance(meta, dict) else []
        pooled = isinstance(meta, dict) and meta.get("pooled_output", None) is not None

        if not torch.is_tensor(tensor):
            lines.append(f"[{index}] non-tensor conditioning payload: {type(tensor).__name__}")
            continue

        work = tensor.detach()
        finite = torch.isfinite(work)
        nan_count = torch.isnan(work).sum().item()
        inf_count = torch.isinf(work).sum().item()
        token_norms = torch.linalg.vector_norm(work.float(), dim=-1)
        profile, profile_source, profile_warning = _resolve_model_profile(work, meta, profile_request)

        lines.extend(
            [
                f"[{index}] shape: {tuple(work.shape)}; dtype: {work.dtype}; device: {work.device}",
                f"[{index}] tokens: {work.shape[1] if work.ndim >= 2 else 'n/a'}; channels: {work.shape[-1] if work.ndim >= 1 else 'n/a'}",
                f"[{index}] meta_keys: {meta_keys}; pooled_output: {pooled}",
                f"[{index}] model_profile: {profile}; profile_source: {profile_source}",
                (
                    f"[{index}] token_norm min/mean/std/max: "
                    f"{token_norms.min().item():.6g} / {token_norms.mean().item():.6g} / "
                    f"{token_norms.std(unbiased=False).item():.6g} / {token_norms.max().item():.6g}"
                ),
                f"[{index}] finite: {bool(finite.all().item())}; nan: {nan_count}; inf: {inf_count}",
            ]
        )
        if index in rms_pairs:
            before, after = rms_pairs[index]
            lines.append(f"[{index}] rms before/after: {before:.6g} / {after:.6g}")
        if profile_warning:
            lines.append(f"[{index}] warning: {profile_warning}")

    return "\n".join(lines)


def _clone_conditioning(conditioning):
    cloned = []
    for tensor, meta in conditioning:
        cloned.append([tensor.clone() if torch.is_tensor(tensor) else tensor, deepcopy(meta)])
    return cloned


def _rms(tensor):
    return torch.sqrt(torch.mean(tensor.float() * tensor.float()).clamp_min(1e-12))


def _resolve_model_profile(tensor, meta, profile_request):
    if profile_request != "auto":
        return profile_request, "explicit", ""

    channels = tensor.shape[-1] if torch.is_tensor(tensor) and tensor.ndim >= 1 else None
    if channels == 768:
        return "sd15_clip", "auto", ""

    if channels == 2048:
        if _looks_like_sdxl_metadata(meta):
            return "sdxl_clip", "auto", ""
        return (
            "generic",
            "fallback",
            "2048-channel conditioning has no SDXL metadata/pooled_output; using conservative generic scaling.",
        )

    if channels in (2560, 4096):
        return "t5", "auto_conservative", f"common T5-like channel dim detected ({channels})."

    return "generic", "fallback", f"unrecognized channel dim ({channels}); using conservative generic scaling."


def _looks_like_sdxl_metadata(meta):
    if not isinstance(meta, dict):
        return False
    if meta.get("pooled_output", None) is not None:
        return True
    sdxl_keys = {"width", "height", "crop_w", "crop_h", "target_width", "target_height"}
    return bool(sdxl_keys.intersection(meta.keys()))


def _center_channels(tensor, amount):
    amount = float(max(0.0, min(1.0, amount)))
    if amount <= 0.0:
        return tensor
    centered = tensor - tensor.mean(dim=-1, keepdim=True)
    return torch.lerp(tensor, centered, amount)


def _adjust_token_contrast(tensor, amount):
    amount = float(max(-0.95, min(1.0, amount)))
    if abs(amount) <= 1e-8 or tensor.ndim < 2:
        return tensor

    token_norms = torch.linalg.vector_norm(tensor, dim=-1, keepdim=True)
    mean_norm = token_norms.mean(dim=1, keepdim=True)
    target_norms = mean_norm + (token_norms - mean_norm) * (1.0 + amount)
    target_norms = target_norms.clamp_min(1e-6)
    scale = torch.where(token_norms > 1e-6, target_norms / token_norms.clamp_min(1e-6), torch.ones_like(token_norms))
    return tensor * scale


def _smooth_token_norms(tensor, amount):
    amount = float(max(0.0, min(1.0, amount)))
    if amount <= 0.0 or tensor.ndim < 3 or tensor.shape[1] < 2:
        return tensor

    token_norms = torch.linalg.vector_norm(tensor, dim=-1, keepdim=True)
    norms_1d = token_norms.squeeze(-1).unsqueeze(1)
    padded = F.pad(norms_1d, (1, 1), mode="replicate")
    smoothed = F.avg_pool1d(padded, kernel_size=3, stride=1).squeeze(1).unsqueeze(-1)
    target_norms = torch.lerp(token_norms, smoothed, amount).clamp_min(1e-6)
    scale = torch.where(token_norms > 1e-6, target_norms / token_norms.clamp_min(1e-6), torch.ones_like(token_norms))
    return tensor * scale


def _tame_spikes(tensor, amount):
    amount = float(max(0.0, min(1.0, amount)))
    if amount <= 0.0 or tensor.ndim < 2:
        return tensor

    token_norms = torch.linalg.vector_norm(tensor, dim=-1, keepdim=True)
    mean_norm = token_norms.mean(dim=1, keepdim=True)
    std_norm = token_norms.std(dim=1, keepdim=True, unbiased=False)
    ceiling = mean_norm + std_norm
    clamped_norms = torch.minimum(token_norms, ceiling.clamp_min(1e-6))
    target_norms = torch.lerp(token_norms, clamped_norms, amount).clamp_min(1e-6)
    scale = torch.where(token_norms > 1e-6, target_norms / token_norms.clamp_min(1e-6), torch.ones_like(token_norms))
    return tensor * scale


def _apply_preset(tensor, preset, strength, model_profile):
    strength = float(max(0.0, min(2.0, strength)))
    if preset == "neutral_report_only" or strength <= 0.0:
        return tensor

    if preset not in PRESET_PARAMS:
        raise ValueError(f"Unknown Oneko conditioning preset: {preset}")

    params = PRESET_PARAMS[preset]
    scales = PROFILE_SCALES.get(model_profile, PROFILE_SCALES["generic"])
    tensor = _center_channels(tensor, params["center"] * scales["center"] * strength)
    tensor = _smooth_token_norms(tensor, params["smooth"] * scales["smooth"] * strength)
    tensor = _tame_spikes(tensor, params["spike"] * scales["spike"] * strength)
    tensor = _adjust_token_contrast(tensor, params["contrast"] * scales["contrast"] * strength)
    return tensor


class OnekoConditioningAnalyzer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning": ("CONDITIONING",)}}

    FUNCTION = "analyze"
    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "report")
    CATEGORY = "Oneko/03 Conditioning/Inspect"
    DESCRIPTION = "Reports tensor shape, metadata, token norms, and NaN/Inf status for a conditioning."

    def analyze(self, conditioning):
        return (conditioning, _conditioning_report(conditioning, "Oneko Conditioning Analyzer"))


class OnekoConditioningAdjust:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "preset": (PRESETS, {"default": "literal_detail"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "preserve_magnitude": ("BOOLEAN", {"default": True}),
                "enabled": ("BOOLEAN", {"default": True}),
                "model_profile": (MODEL_PROFILES, {"default": "auto"}),
            }
        }

    FUNCTION = "adjust"
    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "report")
    CATEGORY = "Oneko/03 Conditioning/Transform"
    DESCRIPTION = "Deterministic preset-based conditioning reshaping for gentle detail, balance, and spike control."

    def adjust(self, conditioning, preset, strength, preserve_magnitude, enabled, model_profile="auto"):
        output = _clone_conditioning(conditioning)
        if not enabled or preset == "neutral_report_only" or float(strength) <= 0.0:
            report = _conditioning_report(
                output,
                f"Oneko Conditioning Adjust: bypassed ({preset})",
                model_profile,
            )
            return (output, report)

        rms_pairs = {}
        for index, entry in enumerate(output):
            tensor = entry[0]
            if not torch.is_tensor(tensor):
                continue

            original_dtype = tensor.dtype
            original_rms = _rms(tensor)
            profile, _, _ = _resolve_model_profile(tensor, entry[1], model_profile)
            work = tensor.float()
            work = _apply_preset(work, preset, strength, profile)

            if preserve_magnitude:
                adjusted_rms = _rms(work)
                if adjusted_rms > 0:
                    work = work * (original_rms / adjusted_rms)

            output[index][0] = work.to(dtype=original_dtype)
            rms_pairs[index] = (original_rms.item(), _rms(output[index][0]).item())

        report = _conditioning_report(
            output,
            f"Oneko Conditioning Adjust: {preset}; strength={float(strength):.3g}",
            model_profile,
            rms_pairs,
        )
        return (output, report)


NODE_CLASS_MAPPINGS = {
    "OnekoConditioningAnalyzer": OnekoConditioningAnalyzer,
    "OnekoConditioningAdjust": OnekoConditioningAdjust,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnekoConditioningAnalyzer": "Conditioning Analyzer (Oneko)",
    "OnekoConditioningAdjust": "Conditioning Adjust (Oneko)"
}
