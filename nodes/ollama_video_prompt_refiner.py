import hashlib
import json
import re

from .ollama_prompt_refiner import (
    DEFAULT_FALLBACK_MODE,
    DEFAULT_OLLAMA_CONTEXT_LENGTH,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    FALLBACK_MODES,
    NEGATIVE_BASELINES,
    OLLAMA_CONTEXT_LENGTH_CHOICES,
    _available_ollama_models,
    _extract_json_object,
    _normalize_context_length,
    _request_ollama,
    _unload_after_run,
)


VIDEO_TARGET_PROFILES = ("minimax_h3", "wan2_2_video")
DEFAULT_VIDEO_TARGET_PROFILE = "minimax_h3"
VIDEO_PIPELINE_MODES = ("single", "review")
DEFAULT_VIDEO_PIPELINE_MODE = "single"
VIDEO_CREATIVITY_MODES = ("faithful", "balanced", "cinematic")
DEFAULT_VIDEO_CREATIVITY_MODE = "balanced"
H3_SECTION_TARGET_WORDS = 100
H3_SECTION_MIN_WORDS = 60
VIDEO_COMPILER_NUM_PREDICT = 2600
VIDEO_SECTION_KEYS = ("scene", "character", "action", "camera", "visual_style", "audio")
VIDEO_RESPONSE_KEYS = (*VIDEO_SECTION_KEYS, "negative", "report")
H3_SECTION_LABELS = {
    "scene": "Scene",
    "character": "Character",
    "action": "Action",
    "camera": "Camera",
    "visual_style": "Visual Style",
    "audio": "Audio",
}
H3_NEGATIVE_BASELINE = (
    "worst quality",
    "low quality",
    "blurry details",
    "flicker",
    "temporal jitter",
    "identity drift",
    "abrupt motion",
    "frozen motion",
    "inconsistent character",
    "inconsistent limbs",
    "distorted mouth movement",
    "lip-sync mismatch",
    "clipped speech",
    "background noise",
    "unwanted dialogue",
    "subtitles",
    "watermark",
    "compression artifacts",
)

VIDEO_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        **{
            key: {
                "type": "string",
                "description": "Focused English video-production prose, ideally 90 to 120 words.",
            }
            for key in VIDEO_SECTION_KEYS
        },
        "negative": {
            "type": "string",
            "description": "A concise comma-separated list of at most 24 failure terms.",
        },
        "report": {
            "type": "string",
            "description": "One brief sentence describing the substantive refinement.",
        },
    },
    "required": list(VIDEO_RESPONSE_KEYS),
    "additionalProperties": False,
}

VIDEO_REVIEW_KEYS = (
    "all_required_preserved",
    "single_shot_consistent",
    "camera_consistent",
    "quotes_preserved",
    "needs_correction",
    "issues",
    "summary",
)
VIDEO_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "all_required_preserved": {"type": "boolean"},
        "single_shot_consistent": {"type": "boolean"},
        "camera_consistent": {"type": "boolean"},
        "quotes_preserved": {"type": "boolean"},
        "needs_correction": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": list(VIDEO_REVIEW_KEYS),
    "additionalProperties": False,
}

VIDEO_LANGUAGE_SCHEMA = {
    "type": "object",
    "properties": {key: {"type": "string"} for key in VIDEO_SECTION_KEYS},
    "required": list(VIDEO_SECTION_KEYS),
    "additionalProperties": False,
}

VIDEO_SYSTEM_INSTRUCTIONS = """You are a specialist writer and director for AI video-generation prompts.
Return only one valid JSON object with exactly these string keys:
scene, character, action, camera, visual_style, audio, negative, report.
Write production descriptions in natural English, translating German or mixed-language prose faithfully.
Preserve every double-quoted string exactly, including its original language, spelling, punctuation, and capitalization.
Treat double-quoted text in Action or Audio as spoken dialogue. Keep the same spoken line in both sections for MiniMax H3.
Keep the supplied six fields semantically separate. Preserve the main subject, requested action, setting, style, camera intent, quantities, negations, and relationships.
Unless faithful mode is requested you must substantially rewrite and enrich the supplied notes instead of copying or merely concatenating them.
You may infer compatible camera behavior, atmosphere, physical secondary motion, lighting interaction, sound design, or ambience when it clearly supports the source.
Never invent a new main character, replace the central motif, add unrelated objects, add cuts, or turn one shot into a montage.
Complete every required JSON field, close the JSON object, avoid repetition, and keep negative and report concise.
Return no markdown or explanation outside JSON."""

VIDEO_REVIEW_SYSTEM_INSTRUCTIONS = """You are a strict AI-video prompt reviewer.
Compare the compiled JSON with the six source fields and target-profile rules.
Check source fidelity, one-shot temporal continuity, camera compatibility, exact preservation of double-quoted text, and model-specific structure.
Flag invented main characters, replaced motifs, missing requested actions, contradictory camera moves, scene cuts, montage language, translated quotes, and altered dialogue.
Return only valid JSON matching the requested review schema. Do not rewrite the prompt."""

VIDEO_LANGUAGE_SYSTEM_INSTRUCTIONS = """You are a faithful translation stage for AI-video production notes.
Translate all German or mixed-language production prose into natural English.
Do not expand shorten curate beautify or move content between fields.
Preserve every double-quoted string exactly in its original language spelling punctuation spacing and capitalization.
Copy already-English prose unchanged.
Return only one valid JSON object with exactly these six string keys: scene character action camera visual_style audio."""

DOUBLE_QUOTED_PATTERN = re.compile(r'"[^"\r\n]*"')
SINGLE_QUOTED_SPEECH_PATTERN = re.compile(r"(?<!\w)'[^'\r\n]{2,240}'(?!\w)")
GERMAN_HINT_WORDS = frozenset(
    (
        "aber", "als", "auf", "aus", "bei", "das", "dass", "dem", "den", "der", "des", "die",
        "durch", "ein", "eine", "einem", "einen", "einer", "er", "für", "gegen", "geht", "hält",
        "im", "ist", "langsam", "mit", "nach", "nicht", "oder", "sich", "sie", "sind", "steht",
        "über", "und", "unter", "vor", "während", "zu", "zur", "zum",
    )
)
STRONG_GERMAN_HINT_WORDS = frozenset(
    (
        "aber", "dass", "durch", "eine", "einem", "einen", "einer", "für", "gegen", "nicht", "oder",
        "sich", "über", "und", "unter", "während", "zur", "zum",
    )
)


def _normalize_video_profile(target_profile):
    profile = str(target_profile).strip().lower()
    return profile if profile in VIDEO_TARGET_PROFILES else DEFAULT_VIDEO_TARGET_PROFILE


def _normalize_video_pipeline(pipeline_mode):
    mode = str(pipeline_mode).strip().lower()
    return mode if mode in VIDEO_PIPELINE_MODES else DEFAULT_VIDEO_PIPELINE_MODE


def _normalize_creativity_mode(creativity_mode):
    mode = str(creativity_mode).strip().lower()
    return mode if mode in VIDEO_CREATIVITY_MODES else DEFAULT_VIDEO_CREATIVITY_MODE


def _normalize_fallback_mode(fallback_mode):
    mode = str(fallback_mode).strip().lower()
    return mode if mode in FALLBACK_MODES else DEFAULT_FALLBACK_MODE


def _clean_prose(value):
    text = str(value or "")
    protected = []

    def replace_quote(match):
        placeholder = f"\x00ONEKO_QUOTE_{len(protected)}\x00"
        protected.append(match.group(0))
        return placeholder

    text = DOUBLE_QUOTED_PATTERN.sub(replace_quote, text)
    text = " ".join(text.split())
    for index, quoted in enumerate(protected):
        text = text.replace(f"\x00ONEKO_QUOTE_{index}\x00", quoted)
    return text


def _source_sections(**kwargs):
    return {key: str(kwargs.get(key, "")).strip() for key in VIDEO_SECTION_KEYS}


def _validate_source(source):
    if not any(str(source.get(key, "")).strip() for key in VIDEO_SECTION_KEYS):
        raise RuntimeError("Ollama Video Prompt Refiner: at least one video section must contain text")


def _source_needs_translation(source):
    return any(_looks_german(source.get(key, "")) for key in VIDEO_SECTION_KEYS)


def _quoted_tokens(value):
    return DOUBLE_QUOTED_PATTERN.findall(str(value))


def _restore_required_quotes(value, required_quotes):
    """Restore exact source dialogue when a model alters, translates, or drops it."""
    required = []
    for token in required_quotes:
        if token not in required:
            required.append(token)
    if not required:
        return str(value)

    text = SINGLE_QUOTED_SPEECH_PATTERN.sub("", str(value))
    present = _quoted_tokens(text)
    missing = [token for token in required if token not in present]
    if not missing:
        return text

    replacements = iter(missing)

    def replace_extra(match):
        token = match.group(0)
        if token in required:
            return token
        return next(replacements, "")

    text = DOUBLE_QUOTED_PATTERN.sub(replace_extra, text)
    still_missing = [token for token in required if token not in text]
    if still_missing:
        text = f"{text} Spoken dialogue: {' '.join(still_missing)}"
    return _clean_prose(text)


def _looks_german(value):
    unquoted = DOUBLE_QUOTED_PATTERN.sub("", str(value)).casefold()
    if re.search(r"[äöüß]", unquoted):
        return True
    words = re.findall(r"[^\W\d_]+", unquoted, re.UNICODE)
    if any(word in STRONG_GERMAN_HINT_WORDS for word in words):
        return True
    return len({word for word in words if word in GERMAN_HINT_WORDS}) >= 2


def _validate_english_conversion(source, values):
    untranslated = [
        key
        for key in VIDEO_SECTION_KEYS
        if _looks_german(values.get(key, ""))
    ]
    if untranslated:
        raise ValueError("production prose was not translated to English: " + ", ".join(untranslated))


def _word_count(value):
    return len(re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", str(value), re.UNICODE))


def _validate_h3_section_lengths(values):
    invalid = []
    for key in VIDEO_SECTION_KEYS:
        count = _word_count(values.get(key, ""))
        if count < H3_SECTION_MIN_WORDS:
            invalid.append(f"{key}={count}")
    if invalid:
        raise ValueError(
            f"MiniMax H3 sections must each contain at least {H3_SECTION_MIN_WORDS} words: "
            + ", ".join(invalid)
        )


def _validate_quotes(source, values, target_profile):
    for key in VIDEO_SECTION_KEYS:
        missing = [token for token in _quoted_tokens(source.get(key, "")) if token not in values[key]]
        if missing:
            raise ValueError(f"{key} changed or omitted quoted text: {', '.join(missing)}")

    if target_profile == "minimax_h3":
        dialogue_tokens = []
        for key in ("action", "audio"):
            for token in _quoted_tokens(source.get(key, "")):
                if token not in dialogue_tokens:
                    dialogue_tokens.append(token)
        for token in dialogue_tokens:
            if token not in values["action"] or token not in values["audio"]:
                raise ValueError(f"spoken dialogue must remain in both action and audio: {token}")


def _validate_video_result(data, source, target_profile, enforce_h3_minimum=True):
    if not isinstance(data, dict):
        raise ValueError("video JSON root is not an object")
    missing = [key for key in VIDEO_RESPONSE_KEYS if key not in data]
    unexpected = [key for key in data if key not in VIDEO_RESPONSE_KEYS]
    if missing:
        raise ValueError(f"video JSON missing keys: {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"video JSON has unexpected keys: {', '.join(unexpected)}")

    values = {}
    for key in VIDEO_RESPONSE_KEYS:
        if not isinstance(data[key], str):
            raise ValueError(f"{key} must be a string")
        values[key] = _clean_prose(data[key])

    for key in VIDEO_SECTION_KEYS:
        values[key] = _restore_required_quotes(
            values[key],
            _quoted_tokens(source.get(key, "")),
        )
    if target_profile == "minimax_h3":
        dialogue_quotes = []
        for key in ("action", "audio"):
            for token in _quoted_tokens(source.get(key, "")):
                if token not in dialogue_quotes:
                    dialogue_quotes.append(token)
        for key in ("action", "audio"):
            values[key] = _restore_required_quotes(values[key], dialogue_quotes)
    if not any(values[key] for key in VIDEO_SECTION_KEYS):
        raise ValueError("video JSON contains no usable positive sections")
    _validate_quotes(source, values, target_profile)
    _validate_english_conversion(source, values)
    if target_profile == "minimax_h3" and enforce_h3_minimum:
        _validate_h3_section_lengths(values)
    return values


def _validate_language_result(data, source):
    if not isinstance(data, dict):
        raise ValueError("language JSON root is not an object")
    missing = [key for key in VIDEO_SECTION_KEYS if key not in data]
    unexpected = [key for key in data if key not in VIDEO_SECTION_KEYS]
    if missing:
        raise ValueError(f"language JSON missing keys: {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"language JSON has unexpected keys: {', '.join(unexpected)}")

    translated = {}
    for key in VIDEO_SECTION_KEYS:
        if not isinstance(data[key], str):
            raise ValueError(f"language field {key} must be a string")
        translated[key] = _clean_prose(data[key])
        source_quotes = _quoted_tokens(source.get(key, ""))
        translated[key] = _restore_required_quotes(translated[key], source_quotes)
        missing_quotes = [token for token in source_quotes if token not in translated[key]]
        if missing_quotes:
            raise ValueError(f"language field {key} changed quoted text: {', '.join(missing_quotes)}")
        if bool(str(source.get(key, "")).strip()) != bool(translated[key]):
            raise ValueError(f"language stage changed whether {key} is empty")
    _validate_english_conversion(source, translated)
    return translated


def _dedupe_negative(value, baseline):
    terms = [part.strip() for part in re.split(r"[,;\r\n]+", str(value)) if part.strip()]
    seen = {term.casefold() for term in terms}
    for term in baseline:
        if term.casefold() not in seen:
            terms.append(term)
            seen.add(term.casefold())
    return ", ".join(terms)


def _append_report(report, detail):
    report = _clean_prose(report)
    detail = _clean_prose(detail)
    return " ".join(part for part in (report, detail) if part)


def _sync_fallback_dialogue(source):
    source = dict(source)
    dialogue_tokens = []
    for key in ("action", "audio"):
        for token in _quoted_tokens(source.get(key, "")):
            if token not in dialogue_tokens:
                dialogue_tokens.append(token)
    if not dialogue_tokens:
        return source

    for token in dialogue_tokens:
        if token not in source["action"]:
            source["action"] = " ".join(part for part in (source["action"], f"Spoken dialogue: {token}.") if part)
        if token not in source["audio"]:
            source["audio"] = " ".join(part for part in (source["audio"], f"Spoken dialogue: {token}. No other dialogue.") if part)
    return source


def _local_fallback(source, target_profile, reason):
    values = {key: _clean_prose(source.get(key, "")) for key in VIDEO_SECTION_KEYS}
    if target_profile == "minimax_h3":
        values = _sync_fallback_dialogue(values)
        baseline = H3_NEGATIVE_BASELINE
    else:
        baseline = NEGATIVE_BASELINES["wan2_2_video"]
    values["negative"] = _dedupe_negative("", baseline)
    values["report"] = f"Local source-preserving fallback used: {_clean_prose(reason)}"
    return values


def _assemble_output(values, target_profile):
    values = dict(values)
    if target_profile == "minimax_h3":
        values["negative"] = _dedupe_negative(values.get("negative", ""), H3_NEGATIVE_BASELINE)
        blocks = [
            f"[{H3_SECTION_LABELS[key]}]\n{values[key]}"
            for key in VIDEO_SECTION_KEYS
            if values.get(key)
        ]
        prompt = "\n\n".join(blocks)
    else:
        values["negative"] = _dedupe_negative(
            values.get("negative", ""), NEGATIVE_BASELINES["wan2_2_video"]
        )
        prompt = " ".join(
            values[key]
            for key in ("character", "action", "scene", "camera", "visual_style")
            if values.get(key)
        )
        if values.get("audio"):
            values["report"] = _append_report(
                values.get("report", ""),
                "Audio was excluded because the Wan 2.2 profile is visual-only.",
            )
    return prompt, values["negative"], values.get("report", "")


def _profile_instructions(target_profile):
    if target_profile == "minimax_h3":
        return f"""Target profile: MiniMax H3.
- Return natural detailed production prose for Scene Character Action Camera Visual Style and Audio.
- Write 90 to 120 words in every one of the six sections, targeting approximately {H3_SECTION_TARGET_WORDS} words. Never exceed 140 words in a section. Expand substantially, but stay focused and avoid padding, repetition, or runaway verbosity. Never omit a section.
- Action must describe continuous temporal motion with a clear progression and physically plausible secondary movement.
- Camera must specify one compatible framing/movement plan without cuts or contradictory moves.
- Audio must describe ambience, sound effects, music, voice, delivery, and language when relevant.
- Preserve spoken dialogue exactly in both Action and Audio and explicitly prevent additional dialogue when the source requests only one line.
- Never invent, paraphrase, or translate spoken lines. Do not add any dialogue beyond the exact double-quoted source text.
- Negative must contain no more than 24 compact comma-separated video and audio failure terms. Report must be one brief sentence."""
    return """Target profile: Wan 2.2 TI2V-5B.
- Produce one achievable continuous visual shot without cuts, montage, teleportation, or simultaneous unrelated actions.
- Character and Action define the subject, motion progression, direction, speed, expression, secondary motion, and settled end state.
- Scene keeps environment geometry and environmental motion temporally consistent.
- Camera uses one compatible locked, tracking, panning, tilting, pushing, or orbiting plan.
- Visual Style covers lighting, color, lens behavior, medium, and temporal stability.
- Audio must be empty because this profile is visual-only.
- Negative must focus on flicker, jitter, drift, deformation, frozen motion, camera shake, text, watermark, and compression failures."""


def _creativity_instructions(creativity_mode):
    mode = _normalize_creativity_mode(creativity_mode)
    if mode == "faithful":
        return """Creativity mode: faithful.
- Rewrite for clarity and English fluency while staying close to the supplied amount of detail.
- Fill an empty section only when another source field makes the missing production detail unambiguous."""
    if mode == "cinematic":
        return """Creativity mode: cinematic.
- Act as an experienced film director and sound designer rather than a text concatenator.
- Expand each supplied idea into vivid production-ready direction with concrete staging and temporal progression.
- Add compatible secondary motion, material response, environmental reactions, lighting evolution, lens behavior, ambience, effects, and musical character.
- Prefer two to four concise sentences in substantial sections while keeping one coherent achievable shot.
- Creative additions must support the existing subject and motif instead of introducing a different story."""
    return """Creativity mode: balanced.
- Do not copy or merely join the source phrases. Rewrite them into polished production-ready video direction.
- Enrich each supplied idea with a few concrete compatible details such as secondary motion, physical response, atmosphere, lighting interaction, camera timing, or sound texture.
- Give Action a readable start progression and ending. Give Camera one intentional movement plan. Give Audio a coherent ambience effects and music package when applicable.
- Keep the result concise and grounded in the existing subject while making it visibly more useful than the input notes."""


def _build_compile_prompt(source, target_profile, creativity_mode):
    return f"""Refine the six independent source sections into one coherent model-specific video prompt.
Translate production prose to English but preserve every double-quoted string exactly.
Empty sections may receive only restrained details clearly implied by another source section.

{_profile_instructions(target_profile)}

{_creativity_instructions(creativity_mode)}

Source JSON:
{json.dumps(source, ensure_ascii=False, indent=2)}"""


def _build_language_prompt(source):
    return f"""Translate the six video source fields into English before creative refinement.
Keep every field independent and preserve all double-quoted text exactly.
Return exactly the six requested string keys.

Source JSON:
{json.dumps(source, ensure_ascii=False, indent=2)}"""


def _build_language_repair_prompt(source, invalid_response, error):
    return f"""Repair the invalid translation response into the exact six-field JSON schema.
The previous response failed because: {error}
Translate all production prose to English without expanding it and preserve double-quoted text exactly.

Source JSON:
{json.dumps(source, ensure_ascii=False, indent=2)}

Invalid response:
{invalid_response}"""


def _build_repair_prompt(source, target_profile, creativity_mode, invalid_response, error):
    return f"""Repair the invalid video-refiner response into the exact required JSON schema.
The previous response failed because: {error}
Preserve the source, exact quoted strings, and target-profile requirements.

{_profile_instructions(target_profile)}

{_creativity_instructions(creativity_mode)}

Source JSON:
{json.dumps(source, ensure_ascii=False, indent=2)}

Invalid response:
{invalid_response}"""


def _validate_review(data):
    if not isinstance(data, dict):
        raise ValueError("review JSON root is not an object")
    missing = [key for key in VIDEO_REVIEW_KEYS if key not in data]
    unexpected = [key for key in data if key not in VIDEO_REVIEW_KEYS]
    if missing:
        raise ValueError(f"review JSON missing keys: {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"review JSON has unexpected keys: {', '.join(unexpected)}")
    for key in VIDEO_REVIEW_KEYS[:5]:
        if not isinstance(data[key], bool):
            raise ValueError(f"{key} must be a boolean")
    if not isinstance(data["issues"], list) or not all(isinstance(item, str) for item in data["issues"]):
        raise ValueError("issues must be an array of strings")
    if not isinstance(data["summary"], str):
        raise ValueError("summary must be a string")
    review = dict(data)
    review["issues"] = [item.strip() for item in review["issues"] if item.strip()]
    review["summary"] = review["summary"].strip()
    if review["issues"] or not all(review[key] for key in VIDEO_REVIEW_KEYS[:4]):
        review["needs_correction"] = True
    return review


def _build_review_prompt(source, values, target_profile, creativity_mode):
    return f"""Review the compiled video JSON against its source and target profile.
Do not rewrite it. Return only the review JSON.

{_profile_instructions(target_profile)}

{_creativity_instructions(creativity_mode)}

Source JSON:
{json.dumps(source, ensure_ascii=False, indent=2)}

Compiled JSON:
{json.dumps(values, ensure_ascii=False, indent=2)}"""


def _build_correction_prompt(source, values, review, target_profile, creativity_mode):
    return f"""Correct the compiled video JSON exactly once using the review findings.
Return the full compiler JSON schema, not the review schema.
Do not alter exact double-quoted source text or introduce a new main subject.

{_profile_instructions(target_profile)}

{_creativity_instructions(creativity_mode)}

Source JSON:
{json.dumps(source, ensure_ascii=False, indent=2)}

Compiled JSON:
{json.dumps(values, ensure_ascii=False, indent=2)}

Review JSON:
{json.dumps(review, ensure_ascii=False, indent=2)}"""


class OnekoOllamaVideoPromptRefiner:
    @classmethod
    def INPUT_TYPES(cls):
        available_models = _available_ollama_models()
        default_model = DEFAULT_OLLAMA_MODEL if DEFAULT_OLLAMA_MODEL in available_models else available_models[0]
        required = {}
        for key in VIDEO_SECTION_KEYS:
            required[key] = (
                "STRING",
                {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": True,
                    "defaultInput": True,
                    "tooltip": f"Optional raw {H3_SECTION_LABELS[key]} guidance in English, German, or mixed language.",
                },
            )
        required.update(
            {
                "ollama_url": (
                    "STRING",
                    {"default": DEFAULT_OLLAMA_URL, "multiline": False},
                ),
                "ollama_model": (
                    available_models,
                    {"default": default_model},
                ),
                "target_profile": (
                    VIDEO_TARGET_PROFILES,
                    {"default": DEFAULT_VIDEO_TARGET_PROFILE},
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.45, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
                "timeout_seconds": (
                    "INT",
                    {"default": 120, "min": 1, "max": 600},
                ),
                "context_length": (
                    OLLAMA_CONTEXT_LENGTH_CHOICES,
                    {"default": str(DEFAULT_OLLAMA_CONTEXT_LENGTH)},
                ),
            }
        )
        return {
            "required": required,
            "optional": {
                "creativity_mode": (
                    VIDEO_CREATIVITY_MODES,
                    {
                        "default": DEFAULT_VIDEO_CREATIVITY_MODE,
                        "tooltip": "faithful stays close to the source; balanced enriches it; cinematic adds stronger grounded direction and sound design.",
                    },
                ),
                "pipeline_mode": (
                    VIDEO_PIPELINE_MODES,
                    {"default": DEFAULT_VIDEO_PIPELINE_MODE},
                ),
                "fallback_mode": (
                    FALLBACK_MODES,
                    {"default": DEFAULT_FALLBACK_MODE},
                ),
                "unload_after_run": (
                    "BOOLEAN",
                    {"default": True},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "negative", "report")
    FUNCTION = "refine"
    CATEGORY = "Oneko/02 Text/Ollama"
    DESCRIPTION = "Translates and expands six source sections with local Ollama into 80-100-word MiniMax H3 sections or a continuous Wan 2.2 visual shot prompt."

    def _request(self, *args, **kwargs):
        try:
            return _request_ollama(*args, **kwargs)
        except RuntimeError as error:
            message = str(error)
            prefix = "Ollama Prompt Refiner: "
            if message.startswith(prefix):
                message = message[len(prefix) :]
            raise RuntimeError(f"Ollama Video Prompt Refiner: {message}") from error

    def _translate_source(self, source, ollama_url, ollama_model, seed, timeout_seconds, context_length, fallback_mode):
        try:
            response = self._request(
                ollama_url,
                ollama_model,
                _build_language_prompt(source),
                seed,
                0.0,
                1.0,
                timeout_seconds,
                context_length,
                output_schema=VIDEO_LANGUAGE_SCHEMA,
                system_instructions=VIDEO_LANGUAGE_SYSTEM_INSTRUCTIONS,
                num_predict=1400,
                reasoning=False,
            )
        except RuntimeError as error:
            if fallback_mode == "continue":
                return source, f"language_transport_continue:{error}"
            raise

        try:
            return _validate_language_result(_extract_json_object(response), source), "language_translated"
        except ValueError as initial_error:
            try:
                repair = self._request(
                    ollama_url,
                    ollama_model,
                    _build_language_repair_prompt(source, response, initial_error),
                    int(seed) + 1,
                    0.0,
                    1.0,
                    timeout_seconds,
                    context_length,
                    output_schema=VIDEO_LANGUAGE_SCHEMA,
                    system_instructions=VIDEO_LANGUAGE_SYSTEM_INSTRUCTIONS,
                    num_predict=1400,
                    reasoning=False,
                )
                return _validate_language_result(_extract_json_object(repair), source), "language_repair"
            except (RuntimeError, ValueError) as repair_error:
                if fallback_mode == "continue":
                    return source, f"language_invalid_continue:{initial_error};{repair_error}"
                raise RuntimeError(
                    f"Ollama Video Prompt Refiner: translation failed twice; initial={initial_error}; repair={repair_error}"
                ) from repair_error

    def _compile(self, source, target_profile, creativity_mode, ollama_url, ollama_model, seed, temperature, top_p, timeout_seconds, context_length, fallback_mode):
        try:
            response = self._request(
                ollama_url,
                ollama_model,
                _build_compile_prompt(source, target_profile, creativity_mode),
                seed,
                temperature,
                top_p,
                timeout_seconds,
                context_length,
                output_schema=VIDEO_OUTPUT_SCHEMA,
                system_instructions=VIDEO_SYSTEM_INSTRUCTIONS,
                num_predict=VIDEO_COMPILER_NUM_PREDICT,
                reasoning=False,
            )
        except RuntimeError as error:
            if fallback_mode == "continue":
                return _local_fallback(source, target_profile, error), "compiler_transport_fallback"
            raise

        try:
            return _validate_video_result(_extract_json_object(response), source, target_profile), "compiler"
        except ValueError as initial_error:
            repair = None
            try:
                repair = self._request(
                    ollama_url,
                    ollama_model,
                    _build_repair_prompt(source, target_profile, creativity_mode, response, initial_error),
                    int(seed) + 1,
                    0.0,
                    1.0,
                    timeout_seconds,
                    context_length,
                    output_schema=VIDEO_OUTPUT_SCHEMA,
                    system_instructions=VIDEO_SYSTEM_INSTRUCTIONS,
                    num_predict=VIDEO_COMPILER_NUM_PREDICT,
                    reasoning=False,
                )
                values = _validate_video_result(_extract_json_object(repair), source, target_profile)
                return values, "compiler_repair"
            except (RuntimeError, ValueError) as repair_error:
                if fallback_mode == "strict":
                    raise RuntimeError(
                        f"Ollama Video Prompt Refiner: compiler JSON invalid twice; initial={initial_error}; repair={repair_error}"
                    ) from repair_error
                for candidate, candidate_name in ((repair, "repair"), (response, "compiler")):
                    if not candidate:
                        continue
                    try:
                        values = _validate_video_result(
                            _extract_json_object(candidate),
                            source,
                            target_profile,
                            enforce_h3_minimum=False,
                        )
                    except ValueError:
                        continue
                    values["report"] = _append_report(
                        values.get("report", ""),
                        f"Adaptive mode kept the creative {candidate_name} result despite sections below {H3_SECTION_MIN_WORDS} words.",
                    )
                    return values, "compiler_relaxed_fallback"
                return _local_fallback(
                    source,
                    target_profile,
                    f"compiler JSON invalid twice; initial={initial_error}; repair={repair_error}",
                ), "compiler_validation_fallback"

    def _review(self, source, values, target_profile, creativity_mode, ollama_url, ollama_model, seed, temperature, top_p, timeout_seconds, context_length, fallback_mode):
        try:
            response = self._request(
                ollama_url,
                ollama_model,
                _build_review_prompt(source, values, target_profile, creativity_mode),
                int(seed) + 2,
                0.1,
                1.0,
                timeout_seconds,
                context_length,
                output_schema=VIDEO_REVIEW_SCHEMA,
                system_instructions=VIDEO_REVIEW_SYSTEM_INSTRUCTIONS,
                num_predict=500,
                reasoning=False,
            )
            review = _validate_review(_extract_json_object(response))
        except (RuntimeError, ValueError) as error:
            if fallback_mode == "strict":
                raise RuntimeError(f"Ollama Video Prompt Refiner: review failed: {error}") from error
            values["report"] = _append_report(values.get("report", ""), f"Review skipped: {error}")
            return values, "review_skipped"

        if not review["needs_correction"]:
            values["report"] = _append_report(values.get("report", ""), f"Review passed: {review['summary']}")
            return values, "review_passed"

        try:
            correction = self._request(
                ollama_url,
                ollama_model,
                _build_correction_prompt(source, values, review, target_profile, creativity_mode),
                int(seed) + 3,
                min(float(temperature), 0.25),
                top_p,
                timeout_seconds,
                context_length,
                output_schema=VIDEO_OUTPUT_SCHEMA,
                system_instructions=VIDEO_SYSTEM_INSTRUCTIONS,
                num_predict=VIDEO_COMPILER_NUM_PREDICT,
                reasoning=False,
            )
            corrected = _validate_video_result(_extract_json_object(correction), source, target_profile)
            corrected["report"] = _append_report(
                corrected.get("report", ""), f"Review correction applied: {review['summary']}"
            )
            return corrected, "review_corrected"
        except (RuntimeError, ValueError) as error:
            if fallback_mode == "strict":
                raise RuntimeError(f"Ollama Video Prompt Refiner: correction failed: {error}") from error
            values["report"] = _append_report(
                values.get("report", ""), f"Review requested correction but compiler result was kept: {error}"
            )
            return values, "review_correction_failed"

    def refine(
        self,
        scene,
        character,
        action,
        camera,
        visual_style,
        audio,
        ollama_url,
        ollama_model,
        target_profile,
        seed,
        temperature,
        top_p,
        timeout_seconds,
        context_length=DEFAULT_OLLAMA_CONTEXT_LENGTH,
        creativity_mode=DEFAULT_VIDEO_CREATIVITY_MODE,
        pipeline_mode=DEFAULT_VIDEO_PIPELINE_MODE,
        fallback_mode=DEFAULT_FALLBACK_MODE,
        unload_after_run=True,
    ):
        source = _source_sections(
            scene=scene,
            character=character,
            action=action,
            camera=camera,
            visual_style=visual_style,
            audio=audio,
        )
        _validate_source(source)
        profile = _normalize_video_profile(target_profile)
        creativity = _normalize_creativity_mode(creativity_mode)
        pipeline = _normalize_video_pipeline(pipeline_mode)
        fallback = _normalize_fallback_mode(fallback_mode)
        try:
            canonical_source, language_stage = self._translate_source(
                source,
                ollama_url,
                ollama_model,
                seed,
                timeout_seconds,
                context_length,
                fallback,
            )
            values, compile_stage = self._compile(
                canonical_source,
                profile,
                creativity,
                ollama_url,
                ollama_model,
                int(seed) + 10,
                temperature,
                top_p,
                timeout_seconds,
                context_length,
                fallback,
            )
            review_stage = "review_disabled"
            if pipeline == "review" and not compile_stage.endswith("fallback"):
                values, review_stage = self._review(
                    canonical_source,
                    values,
                    profile,
                    creativity,
                    ollama_url,
                    ollama_model,
                    int(seed) + 10,
                    temperature,
                    top_p,
                    timeout_seconds,
                    context_length,
                    fallback,
                )
            values["report"] = _append_report(
                values.get("report", ""),
                f"profile={profile}; creativity={creativity}; pipeline={pipeline}; stages={language_stage},{compile_stage},{review_stage}.",
            )
            return _assemble_output(values, profile)
        finally:
            _unload_after_run(
                ollama_url,
                ollama_model,
                timeout_seconds,
                bool(unload_after_run),
            )

    @classmethod
    def IS_CHANGED(
        cls,
        scene,
        character,
        action,
        camera,
        visual_style,
        audio,
        ollama_url,
        ollama_model,
        target_profile,
        seed,
        temperature,
        top_p,
        timeout_seconds,
        context_length=DEFAULT_OLLAMA_CONTEXT_LENGTH,
        creativity_mode=DEFAULT_VIDEO_CREATIVITY_MODE,
        pipeline_mode=DEFAULT_VIDEO_PIPELINE_MODE,
        fallback_mode=DEFAULT_FALLBACK_MODE,
        unload_after_run=True,
    ):
        digest = hashlib.sha256()
        for value in (
            scene,
            character,
            action,
            camera,
            visual_style,
            audio,
            ollama_url,
            ollama_model,
            _normalize_video_profile(target_profile),
            int(seed),
            float(temperature),
            float(top_p),
            int(timeout_seconds),
            _normalize_context_length(context_length),
            _normalize_creativity_mode(creativity_mode),
            _normalize_video_pipeline(pipeline_mode),
            _normalize_fallback_mode(fallback_mode),
            bool(unload_after_run),
        ):
            digest.update(str(value).encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()


NODE_CLASS_MAPPINGS = {
    "OnekoOllamaVideoPromptRefiner": OnekoOllamaVideoPromptRefiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnekoOllamaVideoPromptRefiner": "Ollama Video Prompt Refiner (Oneko)"
}
