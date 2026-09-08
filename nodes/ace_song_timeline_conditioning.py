import json
import math
import re

import torch

from .ace_song_variation_director import (
    SECTION_LINE,
    VARIATION_MARKER,
    _clean_header,
    _clean_line,
    _detect_lyrics_language,
    _format_section,
    _parse_sections,
)


AUDIO_CODES_PER_SECOND = 5
LATENTS_PER_AUDIO_CODE = 5
LATENTS_PER_SECOND = AUDIO_CODES_PER_SECOND * LATENTS_PER_AUDIO_CODE
MAX_SECTIONS = 32
OVERRIDE_LINE = re.compile(r"^\s*([SA]\d+)\s*=\s*(\d+(?:[.,]\d+)?)\s*$", re.IGNORECASE)


def _split_header_direction(value):
    header = _clean_line(value).strip("[] ")
    if VARIATION_MARKER in header:
        header, direction = header.split(VARIATION_MARKER, 1)
        return _clean_header(header), _clean_line(direction)
    clean = _clean_header(header)
    return clean, clean


def _sections_from_headered_lyrics(lyrics):
    lines = str(lyrics or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks = []
    header = None
    body = []

    def append_block(raw_header, body_lines):
        section_lyrics = "\n".join(body_lines).strip()
        if raw_header is None and not section_lyrics:
            return
        clean_header, direction = _split_header_direction(raw_header or "Song Opening")
        blocks.append(
            {
                "id": f"S{len(blocks) + 1:02d}",
                "origin": "source",
                "header": clean_header,
                "direction": direction,
                "lyrics": section_lyrics,
            }
        )

    for line in lines:
        match = SECTION_LINE.fullmatch(line)
        if not match:
            body.append(line)
            continue
        append_block(header, body)
        header = match.group(1)
        body = []
    append_block(header, body)
    return blocks


def _sections_from_lyrics(lyrics):
    text = str(lyrics or "")
    if any(SECTION_LINE.fullmatch(line) for line in text.splitlines()):
        return _sections_from_headered_lyrics(text), "structured_lyrics"
    parsed = _parse_sections(text)
    source = "stanza_detection" if len(parsed) > 1 else "lyrics"
    return (
        [
            {
                "id": item["id"],
                "origin": "source",
                "header": item["header"],
                "direction": item["header"],
                "lyrics": item["lyrics"],
            }
            for item in parsed
        ],
        source,
    )


def _sections_from_plan(plan_json):
    text = str(plan_json or "").strip()
    if not text:
        return None
    data = json.loads(text)
    if not isinstance(data, dict) or data.get("status") != "ok":
        raise ValueError("plan_json status is not ok")
    values = data.get("arranged_sections")
    if not isinstance(values, list) or not values:
        raise ValueError("plan_json has no arranged_sections")
    sections = []
    seen = set()
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("arranged_sections contains a non-object value")
        section_id = str(value.get("id", "")).strip().upper()
        if not re.fullmatch(r"[SA]\d+", section_id) or section_id in seen:
            raise ValueError(f"invalid or duplicate arranged section id: {section_id or '<empty>'}")
        header = _clean_header(value.get("header", ""))
        direction = _clean_line(value.get("direction", "")) or header
        sections.append(
            {
                "id": section_id,
                "origin": str(value.get("origin", "source")),
                "header": header,
                "direction": direction,
                "lyrics": str(value.get("lyrics", "")).strip(),
            }
        )
        seen.add(section_id)
    return sections


def _resolve_sections(lyrics, plan_json):
    warnings = []
    try:
        sections = _sections_from_plan(plan_json)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        warnings.append(f"plan_json ignored: {_clean_line(error)}")
        sections = None
    if sections is not None:
        return sections, "director_plan", warnings
    sections, source = _sections_from_lyrics(lyrics)
    return sections, source, warnings


def _coerce_audio_code_batches(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if not isinstance(value, (list, tuple)) or not value:
        return None
    if all(isinstance(item, int) for item in value):
        value = [value]
    batches = []
    for batch in value:
        if isinstance(batch, torch.Tensor):
            batch = batch.detach().cpu().tolist()
        if not isinstance(batch, (list, tuple)) or not batch:
            return None
        if not all(isinstance(code, int) for code in batch):
            raise ValueError("base_conditioning contains non-integer audio codes")
        batches.append(list(batch))
    if len({len(batch) for batch in batches}) != 1:
        raise ValueError("base_conditioning contains a non-rectangular audio-code batch")
    return batches


def _extract_audio_codes(base_conditioning):
    candidates = []
    for item in base_conditioning:
        if not isinstance(item, (list, tuple)) or len(item) != 2 or not isinstance(item[1], dict):
            raise ValueError("base_conditioning is not a valid CONDITIONING value")
        metadata = item[1]
        if "area" in metadata or "mask" in metadata:
            raise ValueError("base_conditioning is already regionalized")
        batches = _coerce_audio_code_batches(metadata.get("audio_codes"))
        if batches:
            candidates.append(batches)
    if not candidates:
        raise ValueError(
            "ACE Song Timeline Conditioning requires generated ACE-Step 1.5 audio_codes; "
            "enable generate_audio_codes and place the node after ACEtricks compression"
        )
    return max(candidates, key=lambda batches: len(batches[0]))


def _parse_duration_overrides(value, section_ids):
    result = {}
    warnings = []
    valid_ids = set(section_ids)
    for line_number, raw_line in enumerate(str(value or "").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = OVERRIDE_LINE.fullmatch(line)
        if not match:
            raise ValueError(f"invalid duration override on line {line_number}: {raw_line}")
        section_id = match.group(1).upper()
        if section_id not in valid_ids:
            raise ValueError(f"unknown section id in duration_overrides: {section_id}")
        if section_id in result:
            raise ValueError(f"duplicate duration override: {section_id}")
        seconds = float(match.group(2).replace(",", "."))
        if seconds <= 0:
            raise ValueError(f"duration override must be positive: {section_id}")
        code_count = max(1, math.floor(seconds * AUDIO_CODES_PER_SECOND + 0.5))
        rounded = code_count / AUDIO_CODES_PER_SECOND
        if not math.isclose(seconds, rounded, abs_tol=1e-9):
            warnings.append(f"{section_id} rounded from {seconds:g}s to {rounded:g}s")
        result[section_id] = code_count
    return result, warnings


def _section_weight(section):
    word_count = len(re.findall(r"[^\W_]+", section["lyrics"], re.UNICODE))
    return max(1.0, word_count / 12.0)


def _allocate_codes(sections, total_codes, minimum_section_seconds, duration_overrides):
    overrides, warnings = _parse_duration_overrides(
        duration_overrides,
        [section["id"] for section in sections],
    )
    min_codes = max(1, math.ceil(float(minimum_section_seconds) * AUDIO_CODES_PER_SECOND - 1e-9))
    allocations = {section_id: count for section_id, count in overrides.items()}
    automatic = [section for section in sections if section["id"] not in allocations]
    fixed_codes = sum(allocations.values())

    if not automatic:
        if fixed_codes != total_codes:
            raise ValueError(
                f"duration_overrides allocate {fixed_codes / AUDIO_CODES_PER_SECOND:g}s, "
                f"but the audio codes contain {total_codes / AUDIO_CODES_PER_SECOND:g}s"
            )
        return [allocations[section["id"]] for section in sections], warnings

    required = fixed_codes + len(automatic) * min_codes
    if required > total_codes:
        raise ValueError(
            "the fixed durations and minimum_section_seconds exceed the available "
            f"{total_codes / AUDIO_CODES_PER_SECOND:g}s"
        )

    remainder = total_codes - required
    weights = [_section_weight(section) for section in automatic]
    weight_total = sum(weights)
    exact_shares = [remainder * weight / weight_total for weight in weights]
    shares = [math.floor(value) for value in exact_shares]
    leftover = remainder - sum(shares)
    order = sorted(
        range(len(automatic)),
        key=lambda index: (exact_shares[index] - shares[index], -index),
        reverse=True,
    )
    for index in order[:leftover]:
        shares[index] += 1
    for section, share in zip(automatic, shares, strict=True):
        allocations[section["id"]] = min_codes + share
    return [allocations[section["id"]] for section in sections], warnings


def _transition_windows(allocations, transition_seconds):
    requested = max(0, math.floor(float(transition_seconds) * AUDIO_CODES_PER_SECOND + 0.5))
    windows = []
    boundary = allocations[0]
    for left, right in zip(allocations, allocations[1:]):
        overlap_codes = min(requested, left, right)
        left_codes = overlap_codes // 2
        right_codes = overlap_codes - left_codes
        windows.append(
            (
                (boundary - left_codes) * LATENTS_PER_AUDIO_CODE,
                (boundary + right_codes) * LATENTS_PER_AUDIO_CODE,
                overlap_codes,
            )
        )
        boundary += right
    return windows


def _region_mask(total_latents, nominal_start, nominal_end, previous_window, next_window):
    area_start = previous_window[0] if previous_window is not None else nominal_start
    area_end = next_window[1] if next_window is not None else nominal_end
    mask = torch.zeros((1, total_latents), dtype=torch.float32, device="cpu")
    mask[:, area_start:area_end] = 1.0
    if previous_window is not None:
        fade_start, fade_end, _ = previous_window
        fade_length = fade_end - fade_start
        if fade_length:
            ramp = 0.5 - 0.5 * torch.cos(torch.linspace(0.0, math.pi, fade_length))
            mask[:, fade_start:fade_end] = ramp
    if next_window is not None:
        fade_start, fade_end, _ = next_window
        fade_length = fade_end - fade_start
        if fade_length:
            ramp = 0.5 - 0.5 * torch.cos(torch.linspace(0.0, math.pi, fade_length))
            mask[:, fade_start:fade_end] = 1.0 - ramp
    return mask, area_start, area_end


def _copy_conditioning(conditioning, strength_factor):
    result = []
    for tensor, metadata in conditioning:
        copied = metadata.copy()
        copied["strength"] = float(copied.get("strength", 1.0)) * float(strength_factor)
        result.append([tensor.clone(), copied])
    return result


def _timeline_payload(status, source, sections, allocations, warnings, total_codes):
    position = 0
    timeline = []
    for section, code_count in zip(sections, allocations, strict=True):
        code_end = position + code_count
        timeline.append(
            {
                "id": section["id"],
                "origin": section["origin"],
                "header": section["header"],
                "direction": section["direction"],
                "start": position / AUDIO_CODES_PER_SECOND,
                "end": code_end / AUDIO_CODES_PER_SECOND,
                "duration": code_count / AUDIO_CODES_PER_SECOND,
                "code_start": position,
                "code_end": code_end,
                "latent_start": position * LATENTS_PER_AUDIO_CODE,
                "latent_end": code_end * LATENTS_PER_AUDIO_CODE,
            }
        )
        position = code_end
    return {
        "status": status,
        "source": source,
        "effective_duration": total_codes / AUDIO_CODES_PER_SECOND,
        "warnings": warnings,
        "sections": timeline,
    }


class OnekoAceSongTimelineConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "base_conditioning": ("CONDITIONING",),
                "tags": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": False, "defaultInput": True},
                ),
                "lyrics": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": False, "defaultInput": True},
                ),
                "base_strength": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "region_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01},
                ),
                "transition_seconds": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.2},
                ),
                "minimum_section_seconds": (
                    "FLOAT",
                    {"default": 4.0, "min": 0.2, "max": 60.0, "step": 0.2},
                ),
                "duration_overrides": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "Optional exact section durations, one per line, for example S01=8 or A01=6.",
                    },
                ),
            },
            "optional": {
                "plan_json": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": False, "defaultInput": True},
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("conditioning", "timeline_json", "report", "duration_seconds")
    FUNCTION = "build_timeline"
    CATEGORY = "Oneko/08 Audio/ACE"
    DESCRIPTION = "Builds coherent time-regional ACE-Step 1.5 conditioning from song sections."

    def build_timeline(
        self,
        clip,
        base_conditioning,
        tags,
        lyrics,
        base_strength,
        region_strength,
        transition_seconds,
        minimum_section_seconds,
        duration_overrides,
        plan_json="",
    ):
        audio_codes = _extract_audio_codes(base_conditioning)
        total_codes = len(audio_codes[0])
        sections, source, warnings = _resolve_sections(lyrics, plan_json)
        if len(sections) > MAX_SECTIONS:
            raise ValueError(f"ACE Song Timeline supports at most {MAX_SECTIONS} sections")
        allocations, allocation_warnings = _allocate_codes(
            sections,
            total_codes,
            minimum_section_seconds,
            duration_overrides,
        )
        warnings.extend(allocation_warnings)
        duration_seconds = total_codes / AUDIO_CODES_PER_SECOND

        if len(sections) == 1:
            result = _copy_conditioning(base_conditioning, 1.0)
            timeline = _timeline_payload("base_only", source, sections, allocations, warnings, total_codes)
            report = (
                f"ACE Song Timeline: one section across {duration_seconds:g}s; "
                "regional encoding skipped and base strength kept at 1.0."
            )
            if warnings:
                report += " " + "; ".join(warnings)
            return result, json.dumps(timeline, ensure_ascii=False, indent=2), report, duration_seconds

        result = _copy_conditioning(base_conditioning, base_strength)
        windows = _transition_windows(allocations, transition_seconds)
        language = _detect_lyrics_language("\n".join(section["lyrics"] for section in sections))
        total_latents = total_codes * LATENTS_PER_AUDIO_CODE
        code_position = 0

        for index, (section, code_count) in enumerate(zip(sections, allocations, strict=True)):
            code_end = code_position + code_count
            nominal_start = code_position * LATENTS_PER_AUDIO_CODE
            nominal_end = code_end * LATENTS_PER_AUDIO_CODE
            previous_window = windows[index - 1] if index else None
            next_window = windows[index] if index < len(windows) else None
            mask, area_start, area_end = _region_mask(
                total_latents,
                nominal_start,
                nominal_end,
                previous_window,
                next_window,
            )
            area_code_start = area_start // LATENTS_PER_AUDIO_CODE
            area_code_end = math.ceil(area_end / LATENTS_PER_AUDIO_CODE)
            section_tags = (
                f"{str(tags).strip()}\n\n"
                f"Current time region ({section['header']}): {section['direction']}"
            ).strip()
            section_lyrics = _format_section(
                section["header"],
                section["direction"],
                section["lyrics"],
            )
            tokens = clip.tokenize(
                section_tags,
                lyrics=section_lyrics,
                duration=code_count / AUDIO_CODES_PER_SECOND,
                language=language,
                generate_audio_codes=False,
            )
            try:
                regional = clip.encode_from_tokens_scheduled(tokens)
            except Exception as error:
                raise RuntimeError(f"failed to encode timeline section {section['id']}: {error}") from error
            sliced_codes = [batch[area_code_start:area_code_end] for batch in audio_codes]
            for tensor, metadata in regional:
                copied = metadata.copy()
                copied.update(
                    {
                        "audio_codes": sliced_codes,
                        "area": (area_end - area_start, area_start),
                        "mask": mask,
                        "mask_strength": 1.0,
                        "set_area_to_bounds": False,
                        "strength": float(copied.get("strength", 1.0)) * float(region_strength),
                    }
                )
                result.append([tensor.clone(), copied])
            code_position = code_end

        requested_transition = max(0, math.floor(float(transition_seconds) * AUDIO_CODES_PER_SECOND + 0.5))
        if any(window[2] < requested_transition for window in windows):
            warnings.append("one or more transitions were shortened to fit neighboring sections")
        timeline = _timeline_payload("ok", source, sections, allocations, warnings, total_codes)
        report = (
            f"ACE Song Timeline: {len(sections)} regions across {duration_seconds:g}s "
            f"from {source}; base {float(base_strength):g}, regions {float(region_strength):g}, "
            f"transition {float(transition_seconds):g}s."
        )
        if warnings:
            report += " " + "; ".join(warnings)
        return result, json.dumps(timeline, ensure_ascii=False, indent=2), report, duration_seconds


NODE_CLASS_MAPPINGS = {
    "OnekoAceSongTimelineConditioning": OnekoAceSongTimelineConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnekoAceSongTimelineConditioning": "ACE Song Timeline Conditioning (Oneko)"
}
