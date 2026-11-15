from copy import deepcopy

import torch
import torch.nn.functional as F

from .conditioning_context import CONDITIONING_CONTEXT_KEY, conditioning_fingerprint


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
    "moss_image",
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

# Start with the existing conservative scaling; this is not a quality optimum.
PROFILE_SCALES["moss_image"] = dict(PROFILE_SCALES["generic"])


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
        if profile == "moss_image":
            try:
                mask = _moss_mask(work, meta)
                lines.append(f"[{index}] MOSS user-token context: editable {sum(mask)}; protected {len(mask) - sum(mask)}")
            except ValueError as error:
                lines.append(f"[{index}] warning: {error}")
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
    context = meta.get(CONDITIONING_CONTEXT_KEY) if isinstance(meta, dict) else None
    moss_context = isinstance(context, dict) and context.get("model_profile") == "moss_image"
    if profile_request != "auto":
        warning = ""
        if moss_context and profile_request != "moss_image":
            warning = "MOSS encoder context requires auto or moss_image for protected adjustment."
        return profile_request, "explicit", warning
    if moss_context:
        return "moss_image", "oneko_encoder", ""

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
        return (
            "t5", "legacy_shape_heuristic",
            f"Ambiguous {channels}-channel conditioning; preserving legacy T5-like scaling, "
            "not identifying the model family. For MOSS use a Oneko encoder with user-token context.",
        )

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


def _moss_mask(tensor, meta):
    context = meta.get(CONDITIONING_CONTEXT_KEY) if isinstance(meta, dict) else None
    guidance = (
        "MOSS Conditioning Adjust requires verified user-token context from a Oneko encoder. "
        "Use CLIP Sculpt Text Encode (intensity 0, normalization none), a Oneko Balancer, "
        "or a Oneko Regional encoder before Adjust."
    )
    if not isinstance(context, dict) or context.get("version") != 1 or context.get("model_profile") != "moss_image":
        raise ValueError(guidance)
    mask = context.get("token_mask")
    if (
        not torch.is_tensor(tensor) or tensor.ndim != 3
        or not isinstance(mask, (tuple, list))
        or any(type(value) is not bool for value in mask)
        or len(mask) != tensor.shape[1]
        or context.get("token_count") != tensor.shape[1]
    ):
        detail = context.get("warning") or "Token context does not match the conditioning shape."
        raise ValueError(f"{guidance} {detail}")
    if context.get("source_fingerprint") != conditioning_fingerprint(tensor):
        raise ValueError(
            "MOSS conditioning was changed after its user-token context was captured. "
            "Place Adjust directly after a Oneko encoder or another protected Adjust, "
            "before mixing or concatenating different prompts."
        )
    return tuple(mask)


def _mask_runs(mask):
    runs = []
    start = None
    for index, active in enumerate((*mask, False)):
        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append((start, index))
            start = None
    return runs


def _adjust_moss_tensor(tensor, mask, preset, strength, preserve_magnitude):
    # Work only on post-encoder user rows. Neither normalization nor smoothing
    # may cross the protected chat/special-token rows.
    output = tensor.clone()
    if not any(mask):
        return output, []
    params = PRESET_PARAMS[preset]
    scales = PROFILE_SCALES["moss_image"]
    strength = float(max(0.0, min(2.0, strength)))
    indices = torch.tensor(mask, dtype=torch.bool, device=tensor.device)
    runs = _mask_runs(mask)
    rms_pairs = []
    for batch_index in range(tensor.shape[0]):
        work = tensor[batch_index:batch_index + 1, indices, :].float()
        before = _rms(work)
        work = _center_channels(work, params["center"] * scales["center"] * strength)
        offset = 0
        for start, end in runs:
            count = end - start
            work[:, offset:offset + count] = _smooth_token_norms(
                work[:, offset:offset + count],
                params["smooth"] * scales["smooth"] * strength,
            )
            offset += count
        work = _tame_spikes(work, params["spike"] * scales["spike"] * strength)
        work = _adjust_token_contrast(work, params["contrast"] * scales["contrast"] * strength)
        if preserve_magnitude:
            work = work * (before / _rms(work))
        converted = work.to(dtype=tensor.dtype)
        output[batch_index:batch_index + 1, indices, :] = converted
        rms_pairs.append((before.item(), _rms(converted).item()))
    return output, rms_pairs


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
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Preset amount. Zero is neutral; all MOSS presets preserve protected chat rows."}),
                "preserve_magnitude": ("BOOLEAN", {"default": True, "tooltip": "Preserve RMS. For MOSS: per image over editable user rows only; not a bypass."}),
                "enabled": ("BOOLEAN", {"default": True}),
                "model_profile": (MODEL_PROFILES, {"default": "auto", "tooltip": "auto recognizes MOSS from Oneko encoder context. Untagged conditioning retains legacy shape heuristics; it cannot reliably identify MOSS."}),
            }
        }

    FUNCTION = "adjust"
    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "report")
    CATEGORY = "Oneko/03 Conditioning/Transform"
    DESCRIPTION = "Deterministic conditioning presets. MOSS from Oneko encoders edits user text only, preserving chat rows and native token lengths."

    def adjust(self, conditioning, preset, strength, preserve_magnitude, enabled, model_profile="auto"):
        output = _clone_conditioning(conditioning)
        if not enabled or preset == "neutral_report_only" or float(strength) <= 0.0:
            report = _conditioning_report(
                output,
                f"Oneko Conditioning Adjust: bypassed ({preset})",
                model_profile,
            )
            return (output, report)

        if preset not in PRESET_PARAMS:
            raise ValueError(f"Unknown Oneko conditioning preset: {preset}")
        if model_profile not in MODEL_PROFILES:
            raise ValueError(f"Unknown Oneko conditioning model profile: {model_profile}")
        protected_reports = []
        rms_pairs = {}
        for index, entry in enumerate(output):
            tensor = entry[0]
            if not torch.is_tensor(tensor):
                continue

            original_dtype = tensor.dtype
            original_rms = _rms(tensor)
            profile, _, _ = _resolve_model_profile(tensor, entry[1], model_profile)
            context_present = isinstance(entry[1], dict) and CONDITIONING_CONTEXT_KEY in entry[1]
            if context_present or profile == "moss_image":
                mask = _moss_mask(tensor, entry[1])
                if profile != "moss_image":
                    raise ValueError("MOSS encoder context requires model_profile auto or moss_image.")
                output[index][0], selected_rms = _adjust_moss_tensor(
                    tensor, mask, preset, strength, preserve_magnitude,
                )
                entry[1][CONDITIONING_CONTEXT_KEY]["source_fingerprint"] = conditioning_fingerprint(output[index][0])
                protected_reports.append(
                    f"[{index}] MOSS user text only; editable: {sum(mask)}; protected: {len(mask) - sum(mask)}; "
                    f"editable RMS per batch before/after: {selected_rms}"
                )
            else:
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
        if protected_reports:
            report += "\n" + "\n".join(protected_reports)
        return (output, report)


NODE_CLASS_MAPPINGS = {
    "OnekoConditioningAnalyzer": OnekoConditioningAnalyzer,
    "OnekoConditioningAdjust": OnekoConditioningAdjust,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnekoConditioningAnalyzer": "Conditioning Analyzer (Oneko)",
    "OnekoConditioningAdjust": "Conditioning Adjust (Oneko)"
}
