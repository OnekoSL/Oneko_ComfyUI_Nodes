import base64
import hashlib
import io
import json
import re
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

try:
    from .ollama_prompt_refiner import (
        DEFAULT_OLLAMA_URL,
        OLLAMA_CONTEXT_LENGTH_CHOICES,
        _available_ollama_models,
        _normalize_context_length,
        _normalize_generate_url,
        _strip_reasoning_blocks,
        _unload_after_run,
    )
except ImportError:
    from ollama_prompt_refiner import (
        DEFAULT_OLLAMA_URL,
        OLLAMA_CONTEXT_LENGTH_CHOICES,
        _available_ollama_models,
        _normalize_context_length,
        _normalize_generate_url,
        _strip_reasoning_blocks,
        _unload_after_run,
    )


DEFAULT_OLLAMA_VISION_MODEL = "user-v4/joycaption-beta"
ALPHA_TWO_OLLAMA_MODEL = "hf.co/Jobaar/Llama-JoyCaption-Alpha-Two-GGUF:F16"
DEFAULT_OLLAMA_CONTEXT_LENGTH = 4096
DEFAULT_RESIZE_LONG_EDGE = 1024
CAPTION_MODES = ("natural_caption", "danbooru_tags", "pony_source", "refiner_seed")
OUTPUT_KEYS = ("caption", "tags", "text_seed", "report", "hiresfix_text")

VISION_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        key: {"type": "string", "minLength": 1}
        for key in OUTPUT_KEYS
    },
    "required": list(OUTPUT_KEYS),
    "additionalProperties": False,
}

CONTROL_TAG_PATTERN = re.compile(
    r"\b(?:score_\d+(?:_up)?|rating_[a-z0-9_]+|style_cluster_\d+)\b,?\s*",
    re.IGNORECASE,
)
ABSENCE_PHRASE_PATTERN = re.compile(
    r"\b(?:no|without|absent)\s+(?:"
    r"camera composition|furniture details|scale texture|people|person|humans?|"
    r"text|watermark|logo|signature|anthropomorphism|clothing|accessories|"
    r"anatomy|pose|expression|location|lighting|camera|background|hands?|"
    r"face|eyes?|hair|objects?|props?|furniture|scenery|nature|animals?|"
    r"vehicles?|buildings?|textures?|shading|fur|leather|fabric|metal|"
    r"scales?|reflections?|linework"
    r")\b",
    re.IGNORECASE,
)
FIELD_LABEL_PATTERN = re.compile(
    r"(?im)^\s*(?:caption|tags?|text_seed|text seed|report|hiresfix_text|hiresfix text|hiresfix|detail_prompt|detail prompt|description|image|prompt)\s*[:\-]\s*"
)


def _available_vision_models():
    models = list(_available_ollama_models(DEFAULT_OLLAMA_URL))
    for model in (DEFAULT_OLLAMA_VISION_MODEL, ALPHA_TWO_OLLAMA_MODEL):
        if model not in models:
            models.insert(0 if model == DEFAULT_OLLAMA_VISION_MODEL else len(models), model)
    return models


def _normalize_caption_mode(caption_mode):
    value = str(caption_mode).strip().lower()
    if value in CAPTION_MODES:
        return value
    return "refiner_seed"


def _pil_resize_long_edge(image_file, resize_long_edge):
    try:
        long_edge = int(resize_long_edge)
    except (TypeError, ValueError):
        long_edge = DEFAULT_RESIZE_LONG_EDGE
    if long_edge <= 0:
        return image_file

    width, height = image_file.size
    current_long_edge = max(width, height)
    if current_long_edge <= long_edge:
        return image_file

    scale = long_edge / float(current_long_edge)
    target_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return image_file.resize(target_size, resampling)


def _image_tensor_to_pil(image):
    if image is None:
        raise RuntimeError("Ollama Vision Captioner: image input is missing")

    first = image[0] if getattr(image, "ndim", 0) == 4 else image
    if hasattr(first, "detach"):
        image_np = first.detach().cpu().numpy()
    else:
        image_np = np.asarray(first)

    if image_np.ndim != 3:
        raise RuntimeError(f"Ollama Vision Captioner: expected image tensor [H,W,C], got shape {image_np.shape}")
    if image_np.shape[-1] == 1:
        image_np = np.repeat(image_np, 3, axis=-1)
    elif image_np.shape[-1] > 3:
        image_np = image_np[..., :3]

    image_u8 = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(image_u8, mode="RGB")


def _image_to_base64_jpeg(image, resize_long_edge=DEFAULT_RESIZE_LONG_EDGE):
    image_file = _image_tensor_to_pil(image)
    original_size = image_file.size
    image_file = _pil_resize_long_edge(image_file, resize_long_edge)
    encoded_size = image_file.size

    buffer = io.BytesIO()
    image_file.save(buffer, format="JPEG", quality=92, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii"), original_size, encoded_size


def _image_batch_size(image):
    shape = getattr(image, "shape", ())
    if len(shape) == 4:
        return int(shape[0])
    return 1


def _clean_text(value):
    text = str(value).strip()
    text = re.sub(r"(?is)^```(?:json)?\s*|\s*```$", "", text).strip()
    text = FIELD_LABEL_PATTERN.sub("", text)
    text = CONTROL_TAG_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip(" ,;\n\t")
    return text


def _clean_seed_text(value):
    text = _clean_text(value)
    text = text.replace("_", " ")
    text = ABSENCE_PHRASE_PATTERN.sub(" ", text)
    text = re.sub(r"[{}[\]\"]+", " ", text)
    text = re.sub(r"\s*,\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,;")
    return text


def _clean_tags(value, caption_mode):
    text = _clean_text(value)
    if _normalize_caption_mode(caption_mode) in ("pony_source", "refiner_seed"):
        return _clean_seed_text(text)

    parts = [
        _clean_seed_text(part).replace(" ", "_")
        for part in re.split(r"[,;\n|]+", text)
        if _clean_seed_text(part)
    ]
    if parts:
        return ", ".join(dict.fromkeys(parts))
    return text


def _has_visual_word(source, words):
    for word in words:
        if re.search(rf"\b{re.escape(word)}\b", source, re.IGNORECASE):
            return True
    return False


def _material_detail_additions(source_text):
    source = f" {_clean_seed_text(source_text).lower()} "
    additions = []

    if _has_visual_word(source, ("fur", "furry", "pelt")):
        if _has_visual_word(source, ("shaggy", "rough", "scruffy", "messy", "wild", "coarse", "dirty", "feral")):
            additions.extend(("shaggy fur", "coarse fur texture"))
        else:
            additions.extend(("fluffy fur", "fine fur strands"))
    elif _has_visual_word(source, ("tail", "ears", "animal", "anthro", "wolf", "fox", "cat", "dog")):
        additions.append("soft fur texture")

    material_rules = (
        (("hair", "mane", "bangs"), ("fine hair strands",)),
        (("feather", "feathers", "wing", "wings"), ("layered feathers",)),
        (("scale", "scales", "dragon", "reptile"), ("crisp scale texture",)),
        (("latex", "rubber"), ("glossy latex highlights",)),
        (("leather", "boots", "belt", "strap"), ("leather grain",)),
        (("metal", "armor", "armour", "cybernetic", "robot", "sword", "blade"), ("metal reflections",)),
        (("fabric", "cloth", "dress", "shirt", "skirt", "uniform", "kimono", "suit"), ("fabric weave",)),
        (("skin", "face", "body"), ("soft skin texture",)),
        (("water", "wet", "rain", "ocean", "river"), ("wet reflections",)),
        (("glass", "window", "crystal", "gem"), ("clear reflective edges",)),
        (("forest", "tree", "trees", "moss", "leaf", "leaves"), ("leaf detail", "moss texture")),
        (("city", "street", "building", "room", "interior", "background"), ("background detail",)),
        (("anime", "illustration", "digital", "pixel", "lineart"), ("clean linework", "refined shading")),
    )
    for words, phrases in material_rules:
        if _has_visual_word(source, words):
            additions.extend(phrases)

    additions.extend(("fine detail", "crisp edges", "balanced lighting"))
    return list(dict.fromkeys(additions))


def _build_hiresfix_text(caption, tags, text_seed, hiresfix_text):
    base = _clean_seed_text(hiresfix_text)
    source = _clean_seed_text(f"{caption} {tags} {text_seed} {base}")
    if not base:
        base = source
    additions = _material_detail_additions(source)

    combined = base
    combined_lower = f" {combined.lower()} "
    for phrase in additions:
        if phrase.lower() not in combined_lower:
            combined = f"{combined} {phrase}".strip()
            combined_lower = f" {combined.lower()} "

    words = combined.split()
    if len(words) > 90:
        combined = " ".join(words[:90])
    return combined or "fine detail crisp edges refined texture balanced lighting"


def _extract_json_object(text):
    text = _strip_reasoning_blocks(str(text).strip())
    if not text:
        raise ValueError("empty response")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("response does not contain a JSON object")

    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as error:
        raise ValueError(str(error)) from error


def _validate_result(data):
    if not isinstance(data, dict):
        raise ValueError("JSON root is not an object")
    missing = [key for key in OUTPUT_KEYS if key not in data]
    if missing:
        raise ValueError(f"missing result keys: {', '.join(missing)}")

    values = {key: str(data[key]).strip() for key in OUTPUT_KEYS}
    empty = [key for key, value in values.items() if not value]
    if empty:
        raise ValueError(f"empty keys: {', '.join(empty)}")
    return values


def _build_mode_instruction(caption_mode):
    caption_mode = _normalize_caption_mode(caption_mode)
    if caption_mode == "natural_caption":
        return (
            "Write caption as two to four natural English sentences, about 35-80 words total. "
            "Write tags as 16-32 comma-separated visible subjects, setting, clothing, materials, actions, colors, lighting, camera, and style cues. "
            "Write text_seed as 25-50 compact visual words suitable for an image prompt refiner. "
            "Write hiresfix_text as a compact detail-pass prompt for improving the same image."
        )
    if caption_mode == "danbooru_tags":
        return (
            "Write tags as the primary output using 24-48 comma-separated Danbooru or booru-style tags; do not stop after only a short summary. "
            "Split compound visual concepts into useful individual tags and include visible subject type, anatomy, pose, expression, clothing, accessories, materials, action, props, background objects, location, lighting, colors, image medium, style, and camera/composition when visible. "
            "Only write positive visible tags; never add absence tags such as no_text, no_people, no_pose, or no_watermark. "
            "Write caption as one or two factual sentences, about 25-55 words total. "
            "Write text_seed as 30-60 space-separated visible concepts in compact English, using spaces instead of underscores, without score, rating, or style_cluster control tags. "
            "Write hiresfix_text as a 25-70 word detail-pass prompt for direct HiResFix use."
        )
    if caption_mode == "pony_source":
        return (
            "Write caption as a factual image description, about 25-55 words total. "
            "Write tags and text_seed as 35-65 space-separated concrete words and short phrases with no commas. "
            "Focus on subject, pose, action, expression, body details, outfit, material, props, scene objects, background objects, camera, color, and lighting. "
            "Write hiresfix_text as a comma-free 25-70 word direct detail prompt for an upscale/refine pass."
        )
    return (
        "Write caption as a compact but rich image description, about 30-70 words total. "
        "Write tags as 20-40 concise visible concepts. "
        "Write text_seed as a rich 40-80 word comma-free visual seed for another prompt refiner: subject, action, pose, expression, body/clothing/materials, props, room or landscape objects, background details, camera, lighting, colors, style, and mood. "
        "Write hiresfix_text as a direct HiResFix detail prompt for the same image."
    )


def _build_generation_prompt(caption_mode, custom_instruction):
    extra = str(custom_instruction).strip() or "(none)"
    return f"""Describe the provided image for an image-generation workflow.

Mode instructions:
{_build_mode_instruction(caption_mode)}

Custom instruction:
{extra}

Rules:
- Return valid JSON only.
- Use exactly these string keys: caption, tags, text_seed, report, hiresfix_text.
- Describe what is visible in the image; do not refuse or moralize.
- Prefer useful visual detail over extremely short answers, but do not invent identities, text, logos, or hidden context.
- For tags and text_seed, expand visible compound concepts into separate useful words instead of returning only a broad summary.
- hiresfix_text must describe the same image plus refinement details for a HiResFix/upscale pass. Add useful visible material texture words, for example fluffy fur, shaggy fur, fine hair strands, glossy latex highlights, leather grain, fabric weave, metal reflections, crisp scale texture, refined shading, clean linework, background detail.
- Only include positive visible concepts in tags and text_seed; do not add no_* or "without ..." absence tags.
- Do not add score_9, rating_* or style_cluster_* control tags.
- Do not include markdown, labels outside JSON, explanations, or code fences.
- report must be one short sentence naming the model task and any important limitation.
"""


def _build_repair_prompt(raw_response):
    return f"""Repair this invalid captioner answer into valid JSON only.
Use exactly these string keys: caption, tags, text_seed, report, hiresfix_text.
Every value must be a non-empty string.
Do not add markdown or commentary.

Invalid answer:
{raw_response}"""


def _request_ollama_caption(
    ollama_url,
    model,
    prompt,
    image_b64,
    seed,
    temperature,
    top_p,
    timeout_seconds,
    context_length,
):
    model_name = str(model).strip() or DEFAULT_OLLAMA_VISION_MODEL
    options = {
        "seed": int(seed),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "num_ctx": _normalize_context_length(context_length),
        "num_predict": 1200,
    }

    payload = {
        "model": model_name,
        "prompt": str(prompt),
        "stream": False,
        "format": VISION_OUTPUT_SCHEMA,
        "options": options,
    }
    if image_b64:
        payload["images"] = [image_b64]

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        _normalize_generate_url(ollama_url),
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=max(1, int(timeout_seconds))) as response:
            response_data = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        hint = " The selected model may not support image input." if image_b64 else ""
        raise RuntimeError(f"Ollama Vision Captioner: Ollama HTTP {error.code}: {body}{hint}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Ollama Vision Captioner: could not reach Ollama at {ollama_url}: {error.reason}") from error
    except TimeoutError as error:
        raise RuntimeError(f"Ollama Vision Captioner: Ollama request timed out after {timeout_seconds} seconds") from error

    try:
        envelope = json.loads(response_data)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Ollama Vision Captioner: invalid Ollama response envelope: {error}") from error

    if envelope.get("error"):
        raise RuntimeError(f"Ollama Vision Captioner: Ollama error: {envelope['error']}")
    if not envelope.get("done", False):
        raise RuntimeError("Ollama Vision Captioner: Ollama response did not finish cleanly")

    return _strip_reasoning_blocks(str(envelope.get("response", "")).strip())


def _fallback_from_raw(raw_response, caption_mode, batch_size, original_size, encoded_size, error_message):
    raw = _clean_text(raw_response)
    caption = raw or "Ollama did not return a usable caption."
    text_seed = _clean_seed_text(raw) or "image caption unavailable"
    tags = _clean_tags(raw, caption_mode) or text_seed
    hiresfix_text = _build_hiresfix_text(caption, tags, text_seed, raw)
    report = (
        f"Built fallback caption text after invalid JSON: {error_message}. "
        f"Used first image of batch size {batch_size}; {original_size[0]}x{original_size[1]} encoded as {encoded_size[0]}x{encoded_size[1]}."
    )
    return _postprocess_result(
        {"caption": caption, "tags": tags, "text_seed": text_seed, "report": report, "hiresfix_text": hiresfix_text},
        caption_mode,
        batch_size,
        original_size,
        encoded_size,
    )


def _postprocess_result(values, caption_mode, batch_size, original_size, encoded_size):
    caption_mode = _normalize_caption_mode(caption_mode)
    caption = _clean_text(values.get("caption", ""))
    tags = _clean_tags(values.get("tags", ""), caption_mode)
    text_seed = _clean_seed_text(values.get("text_seed", ""))

    if not text_seed:
        text_seed = _clean_seed_text(f"{caption} {tags}")
    if not tags:
        tags = _clean_tags(text_seed, caption_mode)
    if not caption:
        caption = _clean_text(text_seed)
    hiresfix_text = _build_hiresfix_text(caption, tags, text_seed, values.get("hiresfix_text", ""))

    report = _clean_text(values.get("report", "Captioned image with Ollama vision model."))
    batch_note = f" Used first image of batch size {batch_size}." if batch_size > 1 else ""
    size_note = f" Image {original_size[0]}x{original_size[1]} encoded as {encoded_size[0]}x{encoded_size[1]}."
    if "batch size" not in report.lower():
        report = f"{report}{batch_note}"
    if "encoded as" not in report.lower():
        report = f"{report}{size_note}"

    return caption, tags, text_seed, report, hiresfix_text


class OnekoOllamaVisionCaptioner:
    @classmethod
    def INPUT_TYPES(cls):
        available_models = _available_vision_models()
        default_model = DEFAULT_OLLAMA_VISION_MODEL if DEFAULT_OLLAMA_VISION_MODEL in available_models else available_models[0]
        return {
            "required": {
                "image": ("IMAGE",),
                "ollama_url": (
                    "STRING",
                    {
                        "default": DEFAULT_OLLAMA_URL,
                        "multiline": False,
                        "tooltip": "Base Ollama URL, usually http://127.0.0.1:11434.",
                    },
                ),
                "ollama_model": (
                    available_models,
                    {
                        "default": default_model,
                        "tooltip": "Local Ollama vision model used to caption the image. The dropdown refreshes from the selected Ollama URL in the browser.",
                    },
                ),
                "caption_mode": (
                    CAPTION_MODES,
                    {
                        "default": "refiner_seed",
                        "tooltip": "Caption style and output formatting profile.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xffffffffffffffff,
                        "control_after_generate": True,
                        "tooltip": "Seed passed to Ollama for repeatable captioning.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Ollama generation temperature. Lower is more deterministic.",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.01,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Ollama nucleus sampling value.",
                    },
                ),
                "timeout_seconds": (
                    "INT",
                    {
                        "default": 180,
                        "min": 1,
                        "max": 900,
                        "tooltip": "Maximum time to wait for each Ollama request.",
                    },
                ),
                "context_length": (
                    OLLAMA_CONTEXT_LENGTH_CHOICES,
                    {
                        "default": str(DEFAULT_OLLAMA_CONTEXT_LENGTH),
                        "tooltip": "Ollama num_ctx context window. Higher values need more VRAM/RAM and may be limited by the selected model.",
                    },
                ),
                "resize_long_edge": (
                    "INT",
                    {
                        "default": DEFAULT_RESIZE_LONG_EDGE,
                        "min": 0,
                        "max": 4096,
                        "step": 64,
                        "tooltip": "Downscale the image so its longest edge is at most this size. Use 0 to keep the original size.",
                    },
                ),
            },
            "optional": {
                "custom_instruction": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "defaultInput": True,
                        "tooltip": "Optional extra captioning instructions.",
                    },
                ),
                "unload_after_run": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Unload the Ollama model after the complete node run so ComfyUI can reclaim RAM and VRAM.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = OUTPUT_KEYS
    FUNCTION = "caption"
    CATEGORY = "Oneko/02 Text/Ollama"
    DESCRIPTION = "Uses a local Ollama vision model to caption one ComfyUI image and produce text for prompt refining."

    def _caption(
        self,
        image,
        ollama_url,
        ollama_model,
        caption_mode,
        seed,
        temperature,
        top_p,
        timeout_seconds,
        context_length=DEFAULT_OLLAMA_CONTEXT_LENGTH,
        resize_long_edge=DEFAULT_RESIZE_LONG_EDGE,
        custom_instruction="",
    ):
        caption_mode = _normalize_caption_mode(caption_mode)
        batch_size = _image_batch_size(image)
        image_b64, original_size, encoded_size = _image_to_base64_jpeg(image, resize_long_edge)
        prompt = _build_generation_prompt(caption_mode, custom_instruction)

        raw_response = _request_ollama_caption(
            ollama_url,
            ollama_model,
            prompt,
            image_b64,
            seed,
            temperature,
            top_p,
            timeout_seconds,
            context_length,
        )

        try:
            values = _validate_result(_extract_json_object(raw_response))
            return _postprocess_result(values, caption_mode, batch_size, original_size, encoded_size)
        except ValueError as first_error:
            try:
                repair_response = _request_ollama_caption(
                    ollama_url,
                    ollama_model,
                    _build_repair_prompt(raw_response),
                    None,
                    int(seed) + 1,
                    0.0,
                    1.0,
                    timeout_seconds,
                    context_length,
                )
            except RuntimeError as repair_error:
                return _fallback_from_raw(
                    raw_response,
                    caption_mode,
                    batch_size,
                    original_size,
                    encoded_size,
                    f"initial={first_error}; repair_request={repair_error}",
                )
            try:
                values = _validate_result(_extract_json_object(repair_response))
                return _postprocess_result(values, caption_mode, batch_size, original_size, encoded_size)
            except ValueError as second_error:
                return _fallback_from_raw(
                    raw_response,
                    caption_mode,
                    batch_size,
                    original_size,
                    encoded_size,
                    f"initial={first_error}; repair={second_error}",
                )

    def caption(
        self,
        image,
        ollama_url,
        ollama_model,
        caption_mode,
        seed,
        temperature,
        top_p,
        timeout_seconds,
        context_length=DEFAULT_OLLAMA_CONTEXT_LENGTH,
        resize_long_edge=DEFAULT_RESIZE_LONG_EDGE,
        custom_instruction="",
        unload_after_run=True,
    ):
        try:
            return self._caption(
                image,
                ollama_url,
                ollama_model,
                caption_mode,
                seed,
                temperature,
                top_p,
                timeout_seconds,
                context_length,
                resize_long_edge,
                custom_instruction,
            )
        finally:
            _unload_after_run(
                ollama_url,
                str(ollama_model).strip() or DEFAULT_OLLAMA_VISION_MODEL,
                timeout_seconds,
                bool(unload_after_run),
            )

    @classmethod
    def IS_CHANGED(
        cls,
        image,
        ollama_url,
        ollama_model,
        caption_mode,
        seed,
        temperature,
        top_p,
        timeout_seconds,
        context_length=DEFAULT_OLLAMA_CONTEXT_LENGTH,
        resize_long_edge=DEFAULT_RESIZE_LONG_EDGE,
        custom_instruction="",
        unload_after_run=True,
    ):
        digest = hashlib.sha256()
        first = image[0] if getattr(image, "ndim", 0) == 4 else image
        if hasattr(first, "detach"):
            image_np = first.detach().cpu().numpy()
        else:
            image_np = np.asarray(first)
        digest.update(np.ascontiguousarray(image_np).tobytes())
        for value in (
            ollama_url,
            ollama_model,
            _normalize_caption_mode(caption_mode),
            int(seed),
            float(temperature),
            float(top_p),
            int(timeout_seconds),
            _normalize_context_length(context_length),
            int(resize_long_edge),
            custom_instruction,
            bool(unload_after_run),
        ):
            digest.update(str(value).encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()


NODE_CLASS_MAPPINGS = {
    "OnekoOllamaVisionCaptioner": OnekoOllamaVisionCaptioner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnekoOllamaVisionCaptioner": "Ollama Vision Captioner (Oneko)"
}
