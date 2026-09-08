import os
import re

import comfy.model_base
import folder_paths
from nodes import CLIPLoader, UNETLoader, VAELoader


WEIGHT_DTYPES = ("default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2")
CLIP_TYPES = (
    "stable_diffusion",
    "stable_cascade",
    "sd3",
    "stable_audio",
    "mochi",
    "ltxv",
    "pixart",
    "cosmos",
    "lumina2",
    "wan",
    "hidream",
    "chroma",
    "ace",
    "omnigen2",
    "qwen_image",
    "hunyuan_image",
    "flux2",
    "ovis",
    "longcat_image",
    "cogvideox",
    "lens",
    "pixeldit",
    "ideogram4",
    "boogu",
    "krea2",
)
CLIP_DEVICES = ("default", "cpu")

PREFERRED_UNET = "ANIMA\\mixanimamerge_v3.safetensors"
PREFERRED_CLIP = "qwen_3_06b_base.safetensors"
PREFERRED_VAE = "qwen-image\\qwen_image_vae.safetensors"


def _model_name(unet_name):
    normalized = unet_name.replace("\\", "/")
    filename = normalized.rsplit("/", 1)[-1]
    return os.path.splitext(filename)[0]


def _filename_model_name(unet_name):
    name = _model_name(unet_name)
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name)
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r"_+", "_", name)
    name = re.sub(r"-+", "-", name)
    name = name.strip(" ._-")
    return name or "model"


def _folder_name(unet_name):
    normalized = unet_name.replace("\\", "/")
    if "/" not in normalized:
        return ""
    return normalized.rsplit("/", 1)[0]


def _preferred_name(names, preferred):
    if preferred in names:
        return preferred
    normalized_preferred = preferred.replace("\\", "/")
    for name in names:
        if name.replace("\\", "/") == normalized_preferred:
            return name
    return names[0] if names else ""


def _clip_type_for_model(model, clip_type):
    if clip_type == "stable_diffusion" and isinstance(model.model, comfy.model_base.Krea2):
        return "krea2"
    return clip_type


def _unet_names():
    return folder_paths.get_filename_list("diffusion_models")


def _clip_names():
    return folder_paths.get_filename_list("text_encoders")


def _vae_names():
    return VAELoader.vae_list(VAELoader)


class OnekoDiffusionClipVaeCyclerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        unet_names = _unet_names()
        clip_names = _clip_names()
        vae_names = _vae_names()
        return {
            "required": {
                "unet_name": (
                    unet_names,
                    {
                        "default": _preferred_name(unet_names, PREFERRED_UNET),
                        "control_after_generate": True,
                        "tooltip": "Diffusion model/UNET to load. The frontend can increment, decrement, randomize, or wrap this combo after queueing.",
                    },
                ),
                "clip_name": (
                    clip_names,
                    {
                        "default": _preferred_name(clip_names, PREFERRED_CLIP),
                        "tooltip": "Text encoder/CLIP to load.",
                    },
                ),
                "vae_name": (
                    vae_names,
                    {
                        "default": _preferred_name(vae_names, PREFERRED_VAE),
                        "tooltip": "VAE to load.",
                    },
                ),
                "weight_dtype": (
                    WEIGHT_DTYPES,
                    {
                        "default": "default",
                        "advanced": True,
                        "tooltip": "Weight dtype passed to the core Load Diffusion Model node.",
                    },
                ),
                "clip_type": (
                    CLIP_TYPES,
                    {
                        "default": "stable_diffusion",
                        "tooltip": "CLIP type passed to the core Load CLIP node.",
                    },
                ),
                "clip_device": (
                    CLIP_DEVICES,
                    {
                        "default": "default",
                        "advanced": True,
                        "tooltip": "Load CLIP normally or force it to CPU.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "MODEL",
        "CLIP",
        "VAE",
        "modelname",
        "filename_modelname",
        "unet_name",
        "clip_name",
        "vae_name",
        "folder",
    )
    FUNCTION = "load_models"
    CATEGORY = "Oneko/01 Loaders"
    DESCRIPTION = "Loads a separate diffusion model, CLIP/text encoder, and VAE in one cycler-friendly wrapper node."

    def load_models(
        self,
        unet_name,
        clip_name,
        vae_name,
        weight_dtype="default",
        clip_type="stable_diffusion",
        clip_device="default",
    ):
        model = UNETLoader().load_unet(unet_name, weight_dtype)[0]
        clip = CLIPLoader().load_clip(
            clip_name, _clip_type_for_model(model, clip_type), clip_device
        )[0]
        vae = VAELoader().load_vae(vae_name)[0]
        return (
            model,
            clip,
            vae,
            _model_name(unet_name),
            _filename_model_name(unet_name),
            unet_name,
            clip_name,
            vae_name,
            _folder_name(unet_name),
        )

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        unet_name,
        clip_name,
        vae_name,
        weight_dtype="default",
        clip_type="stable_diffusion",
        clip_device="default",
    ):
        if folder_paths.get_full_path("diffusion_models", unet_name) is None:
            return "Invalid diffusion model file: {}".format(unet_name)
        if folder_paths.get_full_path("text_encoders", clip_name) is None:
            return "Invalid CLIP/text encoder file: {}".format(clip_name)
        if vae_name not in _vae_names():
            return "Invalid VAE file: {}".format(vae_name)
        if weight_dtype not in WEIGHT_DTYPES:
            return "Invalid weight dtype: {}".format(weight_dtype)
        if clip_type not in CLIP_TYPES:
            return "Invalid CLIP type: {}".format(clip_type)
        if clip_device not in CLIP_DEVICES:
            return "Invalid CLIP device: {}".format(clip_device)
        return True


NODE_CLASS_MAPPINGS = {
    "OnekoDiffusionClipVaeCyclerLoader": OnekoDiffusionClipVaeCyclerLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnekoDiffusionClipVaeCyclerLoader": "Diffusion Model + CLIP + VAE Cycler Loader (Oneko)"
}
