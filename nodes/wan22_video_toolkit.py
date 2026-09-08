import json
import math
import os
from pathlib import Path
import re

import torch
import numpy as np
from PIL import Image

import comfy.latent_formats
import comfy.model_management
import comfy.samplers
import comfy.utils
import folder_paths


MODES = ("text_to_video", "image_to_video")
QUALITY_PRESETS = ("draft", "balanced", "quality", "custom")
ORIENTATIONS = ("landscape", "portrait", "square")
RESIZE_MODES = ("center_crop", "pad")

PRESET_DIMENSIONS = {
    "draft": {
        "landscape": (640, 352),
        "portrait": (352, 640),
        "square": (512, 512),
    },
    "balanced": {
        "landscape": (960, 544),
        "portrait": (544, 960),
        "square": (704, 704),
    },
    "quality": {
        "landscape": (1280, 704),
        "portrait": (704, 1280),
        "square": (896, 896),
    },
}


def _snap_dimension(value):
    value = max(32, int(value))
    return max(32, int(math.floor((value / 32.0) + 0.5)) * 32)


def _wan_frame_count(duration_seconds, fps):
    target = max(1.0, float(duration_seconds) * float(fps))
    intervals = max(0, int(math.floor(((target - 1.0) / 4.0) + 0.5)))
    return intervals * 4 + 1


def _build_settings(mode, quality, orientation, duration_seconds, fps, custom_width, custom_height):
    if mode not in MODES:
        raise ValueError(f"Unsupported Wan mode: {mode}")
    if quality not in QUALITY_PRESETS:
        raise ValueError(f"Unsupported Wan quality preset: {quality}")
    if orientation not in ORIENTATIONS:
        raise ValueError(f"Unsupported Wan orientation: {orientation}")

    fps = float(fps)
    if fps <= 0:
        raise ValueError("Wan FPS must be greater than zero")
    duration_seconds = float(duration_seconds)
    if duration_seconds <= 0:
        raise ValueError("Wan duration must be greater than zero")

    if quality == "custom":
        width = _snap_dimension(custom_width)
        height = _snap_dimension(custom_height)
        if orientation == "landscape" and width < height:
            width, height = height, width
        elif orientation == "portrait" and width > height:
            width, height = height, width
        elif orientation == "square":
            side = _snap_dimension((width + height) / 2.0)
            width = height = side
    else:
        width, height = PRESET_DIMENSIONS[quality][orientation]

    length = _wan_frame_count(duration_seconds, fps)
    actual_duration = length / fps
    return {
        "mode": mode,
        "quality": quality,
        "orientation": orientation,
        "width": width,
        "height": height,
        "length": length,
        "fps": fps,
        "requested_duration": duration_seconds,
        "actual_duration": actual_duration,
    }


def _settings_report(settings):
    ignored = " Connected images are ignored in text_to_video mode." if settings["mode"] == "text_to_video" else " A start image is required."
    return (
        f"Wan 2.2 {settings['mode']}: {settings['width']}x{settings['height']}, "
        f"{settings['length']} frames at {settings['fps']:g} FPS "
        f"({settings['actual_duration']:.3f}s playback; requested {settings['requested_duration']:g}s)."
        f"{ignored}"
    )


def _validate_settings(settings):
    if not isinstance(settings, dict):
        raise RuntimeError("Wan settings must come from OnekoWan22VideoSettings")
    required = {"mode", "width", "height", "length", "fps"}
    missing = sorted(required.difference(settings))
    if missing:
        raise RuntimeError(f"Wan settings are missing: {', '.join(missing)}")
    if settings["mode"] not in MODES:
        raise RuntimeError(f"Invalid Wan mode in settings: {settings['mode']}")
    width, height, length = int(settings["width"]), int(settings["height"]), int(settings["length"])
    if width % 32 or height % 32:
        raise RuntimeError("Wan width and height must be divisible by 32")
    if length < 1 or (length - 1) % 4:
        raise RuntimeError("Wan frame length must satisfy 4n+1")
    return settings


def _prepare_image(image, width, height, resize_mode):
    if resize_mode not in RESIZE_MODES:
        raise RuntimeError(f"Unsupported Wan image resize mode: {resize_mode}")
    if image is None or not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise RuntimeError("Wan I2V start_image must be an IMAGE tensor in BHWC format")

    source_height, source_width = int(image.shape[1]), int(image.shape[2])
    if source_width < 1 or source_height < 1:
        raise RuntimeError("Wan I2V start_image has invalid dimensions")

    channels_first = image[..., :3].movedim(-1, 1)
    if resize_mode == "center_crop":
        prepared = comfy.utils.common_upscale(channels_first, width, height, "lanczos", "center")
        return prepared.movedim(1, -1)

    scale = min(width / source_width, height / source_height)
    scaled_width = max(1, min(width, int(math.floor(source_width * scale + 0.5))))
    scaled_height = max(1, min(height, int(math.floor(source_height * scale + 0.5))))
    resized = comfy.utils.common_upscale(channels_first, scaled_width, scaled_height, "lanczos", "disabled").movedim(1, -1)
    canvas = torch.zeros(
        (resized.shape[0], height, width, resized.shape[-1]),
        dtype=resized.dtype,
        device=resized.device,
    )
    left = (width - scaled_width) // 2
    top = (height - scaled_height) // 2
    canvas[:, top : top + scaled_height, left : left + scaled_width, :] = resized
    return canvas


class OnekoWan22VideoSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (MODES, {"default": "text_to_video"}),
                "quality": (QUALITY_PRESETS, {"default": "balanced"}),
                "orientation": (ORIENTATIONS, {"default": "landscape"}),
                "duration_seconds": ("FLOAT", {"default": 5.0, "min": 0.0625, "max": 600.0, "step": 0.25}),
                "fps": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "custom_width": ("INT", {"default": 960, "min": 32, "max": 16384, "step": 32}),
                "custom_height": ("INT", {"default": 544, "min": 32, "max": 16384, "step": 32}),
            }
        }

    RETURN_TYPES = ("WAN22_VIDEO_SETTINGS", "INT", "INT", "INT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("settings", "width", "height", "length", "fps", "actual_duration", "report")
    FUNCTION = "build"
    CATEGORY = "Oneko/07 Video/Wan 2.2"
    DESCRIPTION = "Creates validated TI2V-5B settings, resolution presets, and a Wan-compatible 4n+1 frame count."

    def build(self, mode, quality, orientation, duration_seconds, fps, custom_width, custom_height):
        settings = _build_settings(mode, quality, orientation, duration_seconds, fps, custom_width, custom_height)
        return (
            settings,
            settings["width"],
            settings["height"],
            settings["length"],
            settings["fps"],
            settings["actual_duration"],
            _settings_report(settings),
        )


class OnekoWan22TI2VLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "settings": ("WAN22_VIDEO_SETTINGS",),
                "resize_mode": (RESIZE_MODES, {"default": "center_crop"}),
            },
            "optional": {"start_image": ("IMAGE",)},
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "report")
    FUNCTION = "create"
    CATEGORY = "Oneko/07 Video/Wan 2.2"
    DESCRIPTION = "Creates a Wan 2.2 TI2V-5B latent and safely switches between text-to-video and image-to-video."

    def create(self, vae, settings, resize_mode="center_crop", start_image=None):
        settings = _validate_settings(settings)
        width, height, length = int(settings["width"]), int(settings["height"]), int(settings["length"])
        latent_length = ((length - 1) // 4) + 1
        latent = torch.zeros(
            [1, 48, latent_length, height // 16, width // 16],
            device=comfy.model_management.intermediate_device(),
        )

        if settings["mode"] == "text_to_video":
            return ({"samples": latent}, _settings_report(settings))
        if start_image is None:
            raise RuntimeError("Wan 2.2 I2V mode requires a connected start_image")

        prepared = _prepare_image(start_image[:length], width, height, resize_mode)
        encoded = vae.encode(prepared)
        if encoded.ndim != 5 or encoded.shape[1] != 48:
            raise RuntimeError(
                f"Wan 2.2 TI2V-5B requires a 48-channel Wan 2.2 VAE latent; received shape {tuple(encoded.shape)}"
            )
        if encoded.shape[-3] > latent.shape[-3]:
            raise RuntimeError("Encoded start image is longer than the requested Wan video latent")

        mask = torch.ones(
            [latent.shape[0], 1, latent_length, latent.shape[-2], latent.shape[-1]],
            device=latent.device,
        )
        latent[:, :, : encoded.shape[-3]] = encoded
        mask[:, :, : encoded.shape[-3]] = 0.0
        latent_format = comfy.latent_formats.Wan22()
        latent = latent_format.process_out(latent) * mask + latent * (1.0 - mask)
        return (
            {"samples": latent, "noise_mask": mask},
            _settings_report(settings) + f" Start image prepared with {resize_mode}.",
        )


def _filename_token(value, fallback):
    value = str(value).replace("\\", "/").rsplit("/", 1)[-1]
    value = re.sub(r"\.(?:safetensors|gguf|ckpt|pt)$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return value or fallback


class OnekoWan22RunManifest:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "settings": ("WAN22_VIDEO_SETTINGS",),
                "model_name": ("STRING", {"default": "wan2.2_ti2v_5B_fp16.safetensors"}),
                "prompt_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "sampling_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpmpp_2m"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "bong_tangent"}),
                "shift": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "positive": ("STRING", {"default": "", "multiline": True}),
                "negative": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("manifest_json", "filename_prefix")
    FUNCTION = "compose"
    CATEGORY = "Oneko/07 Video/Wan 2.2"
    DESCRIPTION = "Builds reproducible Wan run metadata and a filename-safe Save Video prefix."

    def compose(self, settings, model_name, prompt_seed, sampling_seed, steps, cfg, sampler, scheduler, shift, positive, negative):
        settings = _validate_settings(settings)
        manifest = {
            "schema": "oneko.wan22.run.v1",
            "model": str(model_name),
            "mode": settings["mode"],
            "quality": settings.get("quality", "custom"),
            "orientation": settings.get("orientation", "custom"),
            "width": int(settings["width"]),
            "height": int(settings["height"]),
            "frames": int(settings["length"]),
            "fps": float(settings["fps"]),
            "prompt_seed": int(prompt_seed),
            "sampling_seed": int(sampling_seed),
            "sampling": {
                "steps": int(steps),
                "cfg": float(cfg),
                "sampler": str(sampler),
                "scheduler": str(scheduler),
                "shift": float(shift),
            },
            "positive": str(positive),
            "negative": str(negative),
        }
        prefix = (
            f"wan22/{_filename_token(model_name, 'wan22_5b')}_{settings['mode']}_"
            f"{settings['width']}x{settings['height']}_f{settings['length']}_"
            f"ps{int(prompt_seed)}_ss{int(sampling_seed)}"
        )
        return (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2), prefix)


def _build_continuation_plan(settings, extension_count):
    settings = _validate_settings(settings)
    extension_count = int(extension_count)
    if not 1 <= extension_count <= 10:
        raise RuntimeError("Wan continuation extension_count must be between 1 and 10")

    segment_frames = int(settings["length"])
    fps = float(settings["fps"])
    total_frames = segment_frames + extension_count * (segment_frames - 1)
    motion_duration = (total_frames - 1) / fps
    container_duration = total_frames / fps
    estimated_frame_ram_gb = (
        int(settings["width"]) * int(settings["height"]) * 3 * total_frames * 4
    ) / (1024 ** 3)
    return {
        "schema": "oneko.wan22.continuation.plan.v1",
        "extension_count": extension_count,
        "segment_frames": segment_frames,
        "trimmed_frames_per_extension": segment_frames - 1,
        "seam_trim_frames": 1,
        "total_frames": total_frames,
        "fps": fps,
        "motion_duration": motion_duration,
        "container_duration": container_duration,
        "width": int(settings["width"]),
        "height": int(settings["height"]),
        "estimated_frame_ram_gb": estimated_frame_ram_gb,
    }


def _validate_continuation_plan(plan):
    if not isinstance(plan, dict) or plan.get("schema") != "oneko.wan22.continuation.plan.v1":
        raise RuntimeError("Wan continuation plan must come from OnekoWan22ContinuationPlan")
    if not 1 <= int(plan.get("extension_count", 0)) <= 10:
        raise RuntimeError("Wan continuation plan has an invalid extension count")
    return plan


def _continuation_report(plan):
    warnings = []
    if int(plan["extension_count"]) > 3:
        warnings.append("More than three extensions are experimental and may accumulate identity drift.")
    if float(plan["estimated_frame_ram_gb"]) >= 6.0:
        warnings.append("The accumulated decoded frame batch has a high conservative RAM estimate.")
    warning_text = " " + " ".join(warnings) if warnings else ""
    return (
        f"Wan continuation: {plan['extension_count']} extension(s), {plan['total_frames']} final frames at "
        f"{plan['fps']:g} FPS, {plan['motion_duration']:.3f}s motion span, "
        f"{plan['container_duration']:.3f}s container duration, approximately "
        f"{plan['estimated_frame_ram_gb']:.2f} GiB decoded frame RAM.{warning_text}"
    )


class OnekoWan22ContinuationPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "settings": ("WAN22_VIDEO_SETTINGS",),
                "extension_count": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            }
        }

    RETURN_TYPES = (
        "WAN22_CONTINUATION_PLAN",
        "INT",
        "INT",
        "FLOAT",
        "FLOAT",
        "FLOAT",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "plan",
        "extension_count",
        "total_frames",
        "motion_duration",
        "container_duration",
        "estimated_frame_ram_gb",
        "empty_log_json",
        "report",
    )
    FUNCTION = "build"
    CATEGORY = "Oneko/07 Video/Wan 2.2"
    DESCRIPTION = "Plans one to ten sequential Wan extensions and reports final duration and conservative frame RAM use."

    def build(self, settings, extension_count):
        plan = _build_continuation_plan(settings, extension_count)
        return (
            plan,
            plan["extension_count"],
            plan["total_frames"],
            plan["motion_duration"],
            plan["container_duration"],
            plan["estimated_frame_ram_gb"],
            "[]",
            _continuation_report(plan),
        )


class OnekoWan22ContinuationRecord:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("WAN22_CONTINUATION_PLAN",),
                "log_json": ("STRING", {"default": "[]", "multiline": True}),
                "iteration": ("INT", {"default": 0, "min": 0, "max": 9}),
                "vision_seed": ("INT", {"default": 33004, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "prompt_seed": ("INT", {"default": 44005, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "sampling_seed": ("INT", {"default": 55006, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "caption": ("STRING", {"default": "", "multiline": True}),
                "positive": ("STRING", {"default": "", "multiline": True}),
                "negative": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log_json",)
    FUNCTION = "append"
    CATEGORY = "Oneko/07 Video/Wan 2.2"
    DESCRIPTION = "Appends one deterministic continuation iteration to a JSON run log."

    def append(self, plan, log_json, iteration, vision_seed, prompt_seed, sampling_seed, caption, positive, negative):
        plan = _validate_continuation_plan(plan)
        iteration = int(iteration)
        if not 0 <= iteration < int(plan["extension_count"]):
            raise RuntimeError(
                f"Wan continuation iteration {iteration} is outside the planned range 0..{int(plan['extension_count']) - 1}"
            )
        try:
            records = json.loads(str(log_json) or "[]")
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Wan continuation log is not valid JSON: {error}") from error
        if not isinstance(records, list):
            raise RuntimeError("Wan continuation log must be a JSON list")

        record = {
            "iteration": iteration,
            "segment_index": iteration + 1,
            "vision_seed": int(vision_seed),
            "prompt_seed": int(prompt_seed),
            "sampling_seed": int(sampling_seed),
            "caption": str(caption),
            "positive": str(positive),
            "negative": str(negative),
        }
        records = [item for item in records if not isinstance(item, dict) or int(item.get("iteration", -1)) != iteration]
        records.append(record)
        records.sort(key=lambda item: int(item.get("iteration", -1)) if isinstance(item, dict) else -1)
        return (json.dumps(records, ensure_ascii=False, sort_keys=True, indent=2),)


class OnekoWan22ContinuationManifest:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_manifest_json": ("STRING", {"default": "{}", "multiline": True}),
                "plan": ("WAN22_CONTINUATION_PLAN",),
                "log_json": ("STRING", {"default": "[]", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("manifest_json", "filename_prefix")
    FUNCTION = "compose"
    CATEGORY = "Oneko/07 Video/Wan 2.2"
    DESCRIPTION = "Combines the base Wan manifest and all loop records into one final continuation manifest and filename."

    def compose(self, base_manifest_json, plan, log_json):
        plan = _validate_continuation_plan(plan)
        try:
            base_manifest = json.loads(str(base_manifest_json))
            records = json.loads(str(log_json))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Wan continuation manifest input is not valid JSON: {error}") from error
        if not isinstance(base_manifest, dict):
            raise RuntimeError("Wan base manifest must be a JSON object")
        if not isinstance(records, list):
            raise RuntimeError("Wan continuation records must be a JSON list")
        if len(records) != int(plan["extension_count"]):
            raise RuntimeError(
                f"Wan continuation expected {plan['extension_count']} record(s), received {len(records)}"
            )

        manifest = {
            "schema": "oneko.wan22.continuation.v1",
            "base_run": base_manifest,
            "continuation": {
                key: value
                for key, value in plan.items()
                if key != "schema"
            },
            "extensions": records,
        }
        model = base_manifest.get("model", "wan22_5b")
        mode = base_manifest.get("mode", "image_to_video")
        prompt_seed = int(base_manifest.get("prompt_seed", 0))
        sampling_seed = int(base_manifest.get("sampling_seed", 0))
        prefix = (
            f"wan22/{_filename_token(model, 'wan22_5b')}_{mode}_"
            f"{plan['width']}x{plan['height']}_ext{plan['extension_count']}_f{plan['total_frames']}_"
            f"ps{prompt_seed}_ss{sampling_seed}"
        )
        return (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2), prefix)


def _wan_runs_root():
    return Path(folder_paths.get_output_directory()) / "oneko" / "wan_runs"


def _safe_run_id(run_id):
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(run_id or "").strip()).strip("._-")
    if not token:
        raise RuntimeError("Wan segment run_id must not be empty")
    if token in {".", ".."}:
        raise RuntimeError("Wan segment run_id is invalid")
    return token


def _run_dir(run_id):
    root = _wan_runs_root().resolve()
    path = (root / _safe_run_id(run_id)).resolve()
    if root != path and root not in path.parents:
        raise RuntimeError("Wan segment run_id resolves outside the output/oneko/wan_runs directory")
    return path


def _frame_files(frames_dir):
    if not frames_dir.exists():
        return []
    files = sorted(frames_dir.glob("frame_*.png"))
    for index, path in enumerate(files):
        expected = f"frame_{index:06d}.png"
        if path.name != expected:
            raise RuntimeError(
                f"Wan frame sequence is not contiguous at index {index}: expected {expected}, found {path.name}"
            )
    return files


def _load_json(path, fallback):
    if not path.exists():
        return fallback
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Wan segment JSON is invalid: {path}: {error}") from error


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)


def _image_tensor_to_pil(image):
    if not isinstance(image, torch.Tensor) or image.ndim != 3:
        raise RuntimeError("Wan segment image must be one BHWC frame tensor")
    image = image.detach().cpu().float().clamp(0.0, 1.0)
    if image.shape[-1] == 1:
        image = image.repeat(1, 1, 3)
    image = image[..., :3]
    array = (image.numpy() * 255.0 + 0.5).astype("uint8")
    return Image.fromarray(array, mode="RGB")


def _pil_to_image_tensor(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        tensor = torch.from_numpy(np.asarray(image).copy()).float() / 255.0
    return tensor.unsqueeze(0)


def _parse_json_object(text, label):
    if not str(text or "").strip():
        return {}
    try:
        payload = json.loads(str(text))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return payload


def _segment_report(run_id, segment_index, first_frame, last_frame, stored_count, total_frames):
    return (
        f"Wan segment {segment_index} stored for run '{run_id}': {stored_count} frame(s), "
        f"frame_{first_frame:06d}.png..frame_{last_frame:06d}.png, {total_frames} total frame(s)."
    )


class OnekoWan22SegmentStore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "run_id": ("STRING", {"default": "wan_run_001"}),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 999, "step": 1}),
                "fps": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "drop_first_frame": ("BOOLEAN", {"default": False}),
                "overwrite_segment": ("BOOLEAN", {"default": True}),
                "vision_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "prompt_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "sampling_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "original_caption": ("STRING", {"default": "", "multiline": True}),
                "current_caption": ("STRING", {"default": "", "multiline": True}),
                "positive": ("STRING", {"default": "", "multiline": True}),
                "negative": ("STRING", {"default": "", "multiline": True}),
                "run_manifest_json": ("STRING", {"default": "{}", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "INT", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = (
        "run_id",
        "run_dir",
        "next_segment_index",
        "stored_frame_count",
        "total_frame_count",
        "last_frame",
        "segment_manifest_json",
        "report",
    )
    FUNCTION = "store"
    CATEGORY = "Oneko/07 Video/Wan 2.2/Segments"
    OUTPUT_NODE = True
    DESCRIPTION = "Stores one Wan segment as numbered PNG frames, trims duplicate seam frames, and updates run state."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def store(
        self,
        images,
        run_id,
        segment_index,
        fps,
        drop_first_frame,
        overwrite_segment,
        vision_seed,
        prompt_seed,
        sampling_seed,
        original_caption,
        current_caption,
        positive,
        negative,
        run_manifest_json,
    ):
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise RuntimeError("OnekoWan22SegmentStore requires an IMAGE batch in BHWC format")
        if images.shape[0] < 1:
            raise RuntimeError("OnekoWan22SegmentStore received an empty image batch")

        run_id = _safe_run_id(run_id)
        segment_index = int(segment_index)
        fps = float(fps)
        run_dir = _run_dir(run_id)
        frames_dir = run_dir / "frames"
        state_dir = run_dir / "state"
        manifests_dir = run_dir / "manifests"
        run_manifest_path = manifests_dir / "run_manifest.json"

        if segment_index == 0 and bool(overwrite_segment) and run_dir.exists():
            for child in (frames_dir, state_dir, manifests_dir):
                if child.exists():
                    for path in sorted(child.glob("*")):
                        if path.is_file():
                            path.unlink()

        frames_dir.mkdir(parents=True, exist_ok=True)
        state_dir.mkdir(parents=True, exist_ok=True)
        manifests_dir.mkdir(parents=True, exist_ok=True)

        prior_run_manifest = _load_json(run_manifest_path, {})
        existing_files = _frame_files(frames_dir)

        if segment_index == 0:
            expected_start = 0
            base_segment_frames = int(images.shape[0])
            if existing_files and not bool(overwrite_segment):
                raise RuntimeError(f"Wan run '{run_id}' already has frames; enable overwrite_segment or use another run_id")
        else:
            if not prior_run_manifest:
                raise RuntimeError(f"Wan run '{run_id}' has no run manifest; render segment 0 first")
            base_segment_frames = int(prior_run_manifest.get("base_segment_frames", 0))
            if base_segment_frames < 2:
                raise RuntimeError("Wan run manifest has an invalid base_segment_frames value")
            expected_start = base_segment_frames + (segment_index - 1) * (base_segment_frames - 1)

        if bool(overwrite_segment):
            for path in list(frames_dir.glob("frame_*.png")):
                try:
                    frame_index = int(path.stem.rsplit("_", 1)[-1])
                except ValueError:
                    continue
                if frame_index >= expected_start:
                    path.unlink()
            existing_files = _frame_files(frames_dir)

        if len(existing_files) != expected_start:
            raise RuntimeError(
                f"Wan run '{run_id}' expected {expected_start} existing frame(s) before segment {segment_index}, "
                f"found {len(existing_files)}"
            )

        start_offset = 1 if bool(drop_first_frame) else 0
        if int(images.shape[0]) <= start_offset:
            raise RuntimeError("Wan segment has no frames left after drop_first_frame")

        frames_to_store = images[start_offset:]
        first_frame_index = expected_start
        for offset, frame in enumerate(frames_to_store):
            _image_tensor_to_pil(frame).save(frames_dir / f"frame_{first_frame_index + offset:06d}.png")

        total_frames = expected_start + int(frames_to_store.shape[0])
        last_frame_index = total_frames - 1
        last_frame_path = frames_dir / f"frame_{last_frame_index:06d}.png"
        state_last_path = state_dir / "last_frame.png"
        _image_tensor_to_pil(frames_to_store[-1]).save(state_last_path)

        external_run_manifest = _parse_json_object(run_manifest_json, "run_manifest_json")
        segment_manifest = {
            "schema": "oneko.wan22.segment.v1",
            "run_id": run_id,
            "segment_index": segment_index,
            "fps": fps,
            "input_frames": int(images.shape[0]),
            "stored_frames": int(frames_to_store.shape[0]),
            "drop_first_frame": bool(drop_first_frame),
            "first_frame_index": first_frame_index,
            "last_frame_index": last_frame_index,
            "vision_seed": int(vision_seed),
            "prompt_seed": int(prompt_seed),
            "sampling_seed": int(sampling_seed),
            "original_caption": str(original_caption),
            "current_caption": str(current_caption),
            "positive": str(positive),
            "negative": str(negative),
            "source_manifest": external_run_manifest,
            "last_frame_path": str(last_frame_path),
        }
        _write_json(manifests_dir / f"segment_{segment_index:03d}.json", segment_manifest)

        if segment_index == 0 or not prior_run_manifest:
            run_manifest = {
                "schema": "oneko.wan22.segmented_run.v1",
                "run_id": run_id,
                "fps": fps,
                "base_segment_frames": base_segment_frames,
                "frame_width": int(images.shape[2]),
                "frame_height": int(images.shape[1]),
                "original_caption": str(original_caption or current_caption),
                "segments": [],
            }
        else:
            run_manifest = prior_run_manifest
            run_manifest["fps"] = fps

        run_manifest["segments"] = [
            item
            for item in run_manifest.get("segments", [])
            if int(item.get("segment_index", -1)) < segment_index
        ]
        run_manifest["segments"].append(
            {
                "segment_index": segment_index,
                "first_frame_index": first_frame_index,
                "last_frame_index": last_frame_index,
                "stored_frames": int(frames_to_store.shape[0]),
                "drop_first_frame": bool(drop_first_frame),
                "manifest": f"segment_{segment_index:03d}.json",
            }
        )
        run_manifest["total_frames"] = total_frames
        run_manifest["next_segment_index"] = segment_index + 1
        run_manifest["last_frame_path"] = str(state_last_path)
        _write_json(run_manifest_path, run_manifest)

        report = _segment_report(
            run_id,
            segment_index,
            first_frame_index,
            last_frame_index,
            int(frames_to_store.shape[0]),
            total_frames,
        )
        return (
            run_id,
            str(run_dir),
            segment_index + 1,
            int(frames_to_store.shape[0]),
            total_frames,
            _pil_to_image_tensor(state_last_path),
            json.dumps(segment_manifest, ensure_ascii=False, sort_keys=True, indent=2),
            report,
        )


class OnekoWan22SegmentLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "run_id": ("STRING", {"default": "wan_run_001"}),
                "segment_index": ("INT", {"default": -1, "min": -1, "max": 999, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "last_frame",
        "run_id",
        "segment_index",
        "total_frame_count",
        "original_caption",
        "current_caption",
        "run_manifest_json",
        "report",
    )
    FUNCTION = "load"
    CATEGORY = "Oneko/07 Video/Wan 2.2/Segments"
    DESCRIPTION = "Loads the current last frame and manifest state for the next manual Wan continuation segment."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def load(self, run_id, segment_index):
        run_id = _safe_run_id(run_id)
        run_dir = _run_dir(run_id)
        manifests_dir = run_dir / "manifests"
        state_last_path = run_dir / "state" / "last_frame.png"
        run_manifest_path = manifests_dir / "run_manifest.json"
        if not run_manifest_path.exists() or not state_last_path.exists():
            raise RuntimeError(f"Wan run '{run_id}' has no saved state; render segment 0 first")
        run_manifest = _load_json(run_manifest_path, {})
        segments = run_manifest.get("segments", [])
        if not isinstance(segments, list) or not segments:
            raise RuntimeError(f"Wan run '{run_id}' has no stored segments")
        next_segment_index = int(run_manifest.get("next_segment_index", len(segments)))
        chosen_segment_index = next_segment_index if int(segment_index) < 0 else int(segment_index)
        if chosen_segment_index < 1:
            raise RuntimeError("Wan continuation segment_index must be -1 for next, or at least 1")

        last_segment = sorted(segments, key=lambda item: int(item.get("segment_index", -1)))[-1]
        last_manifest = _load_json(manifests_dir / str(last_segment.get("manifest", "")), {})
        original_caption = str(run_manifest.get("original_caption") or last_manifest.get("original_caption") or "")
        current_caption = str(last_manifest.get("current_caption") or "")
        report = (
            f"Wan run '{run_id}' loaded: next segment {chosen_segment_index}, "
            f"{int(run_manifest.get('total_frames', 0))} frame(s) currently stored."
        )
        return (
            _pil_to_image_tensor(state_last_path),
            run_id,
            chosen_segment_index,
            int(run_manifest.get("total_frames", 0)),
            original_caption,
            current_caption,
            json.dumps(run_manifest, ensure_ascii=False, sort_keys=True, indent=2),
            report,
        )


class OnekoWan22FrameSequenceAssembler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "run_id": ("STRING", {"default": "wan_run_001"}),
                "fps": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "filename": ("STRING", {"default": ""}),
                "codec": (("libx264", "mpeg4"), {"default": "libx264"}),
                "quality": ("INT", {"default": 8, "min": 1, "max": 10, "step": 1}),
                "overwrite": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("video_path", "total_frames", "duration", "manifest_json", "report")
    FUNCTION = "assemble"
    CATEGORY = "Oneko/07 Video/Wan 2.2/Segments"
    OUTPUT_NODE = True
    DESCRIPTION = "Assembles a saved Wan PNG frame sequence into one MP4 with bundled imageio-ffmpeg."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def assemble(self, run_id, fps, filename, codec, quality, overwrite):
        import imageio.v2 as imageio

        run_id = _safe_run_id(run_id)
        fps = float(fps)
        run_dir = _run_dir(run_id)
        frames_dir = run_dir / "frames"
        manifests_dir = run_dir / "manifests"
        videos_dir = run_dir / "videos"
        files = _frame_files(frames_dir)
        if not files:
            raise RuntimeError(f"Wan run '{run_id}' has no frames to assemble")

        run_manifest = _load_json(manifests_dir / "run_manifest.json", {})
        expected_total = int(run_manifest.get("total_frames", len(files))) if run_manifest else len(files)
        if len(files) != expected_total:
            raise RuntimeError(
                f"Wan run '{run_id}' expected {expected_total} frame(s) from manifest, found {len(files)}"
            )

        videos_dir.mkdir(parents=True, exist_ok=True)
        safe_name = _filename_token(filename or run_id, run_id)
        video_path = videos_dir / f"{safe_name}.mp4"
        if video_path.exists() and not bool(overwrite):
            raise RuntimeError(f"Wan output video already exists: {video_path}")

        first_size = None
        with imageio.get_writer(
            str(video_path),
            fps=fps,
            codec=str(codec),
            quality=int(quality),
            macro_block_size=1,
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        ) as writer:
            for path in files:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    if first_size is None:
                        first_size = image.size
                    elif image.size != first_size:
                        raise RuntimeError(
                            f"Wan frame size mismatch at {path.name}: expected {first_size}, found {image.size}"
                        )
                    writer.append_data(np.asarray(image))

        duration = len(files) / fps
        assembly_manifest = {
            "schema": "oneko.wan22.assembled_video.v1",
            "run_id": run_id,
            "video_path": str(video_path),
            "total_frames": len(files),
            "fps": fps,
            "duration": duration,
            "codec": str(codec),
            "source_run_manifest": run_manifest,
        }
        _write_json(manifests_dir / "assembled_video.json", assembly_manifest)
        report = f"Wan run '{run_id}' assembled: {len(files)} frame(s), {duration:.3f}s, {video_path}"
        return (
            str(video_path),
            len(files),
            duration,
            json.dumps(assembly_manifest, ensure_ascii=False, sort_keys=True, indent=2),
            report,
        )


NODE_CLASS_MAPPINGS = {
    "OnekoWan22VideoSettings": OnekoWan22VideoSettings,
    "OnekoWan22TI2VLatent": OnekoWan22TI2VLatent,
    "OnekoWan22RunManifest": OnekoWan22RunManifest,
    "OnekoWan22ContinuationPlan": OnekoWan22ContinuationPlan,
    "OnekoWan22ContinuationRecord": OnekoWan22ContinuationRecord,
    "OnekoWan22ContinuationManifest": OnekoWan22ContinuationManifest,
    "OnekoWan22SegmentStore": OnekoWan22SegmentStore,
    "OnekoWan22SegmentLoader": OnekoWan22SegmentLoader,
    "OnekoWan22FrameSequenceAssembler": OnekoWan22FrameSequenceAssembler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnekoWan22VideoSettings": "Wan 2.2 Video Settings (Oneko)",
    "OnekoWan22TI2VLatent": "Wan 2.2 TI2V Latent (Oneko)",
    "OnekoWan22RunManifest": "Wan 2.2 Run Manifest (Oneko)",
    "OnekoWan22ContinuationPlan": "Wan 2.2 Continuation Plan (Oneko)",
    "OnekoWan22ContinuationRecord": "Wan 2.2 Continuation Record (Oneko)",
    "OnekoWan22ContinuationManifest": "Wan 2.2 Continuation Manifest (Oneko)",
    "OnekoWan22SegmentStore": "Wan 2.2 Segment Store (Oneko)",
    "OnekoWan22SegmentLoader": "Wan 2.2 Segment Loader (Oneko)",
    "OnekoWan22FrameSequenceAssembler": "Wan 2.2 Frame Sequence Assembler (Oneko)"
}
