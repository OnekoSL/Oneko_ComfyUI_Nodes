import torch

import comfy.samplers

from .noise_sampler_core import (
    MAX_SEED,
    PREVIEW_METHODS,
    UNIVERSAL_NOISE_PROFILES,
    OnekoEmptyNoise,
    make_noise_generator,
    sample_custom_advanced,
)


class OnekoUniversalNoiseSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "add_noise": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Generate noise from the selected Oneko profile. Disable this to sample with zero noise.",
                    },
                ),
                "noise_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": MAX_SEED,
                        "control_after_generate": True,
                        "tooltip": "Seed used for the selected noise profile.",
                    },
                ),
                "noise_device": (
                    ["auto", "cpu", "cuda"],
                    {
                        "default": "auto",
                        "tooltip": "auto uses CPU for ComfyUI-core-like reproducibility. cuda falls back to CPU when unavailable.",
                    },
                ),
                "noise_profile": (
                    UNIVERSAL_NOISE_PROFILES,
                    {
                        "default": "gaussian",
                        "tooltip": "Basic noise type or composite Oneko profile.",
                    },
                ),
                "noise_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": "Final noise multiplier. For gaussian + auto, 1.0 keeps ComfyUI-core-like behavior.",
                    },
                ),
                "detail_bias": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Only affects composite profiles. Lower values emphasize larger forms; higher values emphasize details.",
                    },
                ),
                "preview_method": (
                    PREVIEW_METHODS,
                    {
                        "default": "default",
                        "tooltip": "Per-node latent preview override. default follows ComfyUI; latent2rgb is lightweight; none disables previews.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "INT")
    RETURN_NAMES = ("output", "denoised_output", "seed")
    FUNCTION = "sample"
    CATEGORY = "Oneko/05 Sampling"
    DESCRIPTION = "SamplerCustomAdvanced-style node with all Oneko basic and composite noise profiles."

    def sample(
        self,
        guider,
        sampler,
        sigmas,
        latent_image,
        add_noise,
        noise_seed,
        noise_device,
        noise_profile="gaussian",
        noise_strength=1.0,
        detail_bias=0.35,
        preview_method="default",
    ):
        if add_noise:
            noise = make_noise_generator(noise_seed, noise_device, noise_profile, noise_strength, detail_bias)
        else:
            noise = OnekoEmptyNoise()

        return sample_custom_advanced(
            guider,
            sampler,
            sigmas,
            latent_image,
            noise,
            noise_seed,
            preview_method=preview_method,
        )


class OnekoUniversalNoiseSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "add_noise": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Generate noise from the selected Oneko profile. Disable this to sample with zero noise.",
                    },
                ),
                "noise_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": MAX_SEED,
                        "control_after_generate": True,
                        "tooltip": "Seed used for the selected noise profile.",
                    },
                ),
                "noise_device": (
                    ["auto", "cpu", "cuda"],
                    {
                        "default": "auto",
                        "tooltip": "auto uses CPU for ComfyUI-core-like reproducibility. cuda falls back to CPU when unavailable.",
                    },
                ),
                "noise_profile": (
                    UNIVERSAL_NOISE_PROFILES,
                    {
                        "default": "gaussian",
                        "tooltip": "Basic noise type or composite Oneko profile.",
                    },
                ),
                "noise_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": "Final noise multiplier. For gaussian + auto, 1.0 keeps ComfyUI-core-like behavior.",
                    },
                ),
                "detail_bias": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Only affects composite profiles. Lower values emphasize larger forms; higher values emphasize details.",
                    },
                ),
                "start_at_step": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 10000,
                        "tooltip": "First sigma step to sample, matching KSampler Advanced start_at_step semantics.",
                    },
                ),
                "end_at_step": (
                    "INT",
                    {
                        "default": 10000,
                        "min": 0,
                        "max": 10000,
                        "tooltip": "Last sigma step boundary. 10000 means continue to the end of the incoming sigmas.",
                    },
                ),
                "return_with_leftover_noise": (
                    ["disable", "enable"],
                    {
                        "default": "disable",
                        "tooltip": "Disable forces full denoise when ending early. Enable preserves leftover noise for a later pass.",
                    },
                ),
                "preview_method": (
                    PREVIEW_METHODS,
                    {
                        "default": "default",
                        "tooltip": "Per-node latent preview override. default follows ComfyUI; latent2rgb is lightweight; none disables previews.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "INT")
    RETURN_NAMES = ("output", "denoised_output", "seed")
    FUNCTION = "sample"
    CATEGORY = "Oneko/05 Sampling"
    DESCRIPTION = "Universal Noise Sampler with KSampler Advanced-style step range controls."

    def sample(
        self,
        guider,
        sampler,
        sigmas,
        latent_image,
        add_noise,
        noise_seed,
        noise_device,
        noise_profile="gaussian",
        noise_strength=1.0,
        detail_bias=0.35,
        start_at_step=0,
        end_at_step=10000,
        return_with_leftover_noise="disable",
        preview_method="default",
    ):
        if add_noise:
            noise = make_noise_generator(noise_seed, noise_device, noise_profile, noise_strength, detail_bias)
        else:
            noise = OnekoEmptyNoise()

        return sample_custom_advanced(
            guider,
            sampler,
            sigmas,
            latent_image,
            noise,
            noise_seed,
            start_at_step,
            end_at_step,
            return_with_leftover_noise,
            preview_method=preview_method,
        )


class OnekoUniversalKSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": MAX_SEED,
                        "control_after_generate": True,
                        "tooltip": "Seed used for the selected Oneko noise profile.",
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 20,
                        "min": 1,
                        "max": 10000,
                        "tooltip": "Number of denoising steps.",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 7.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                        "tooltip": "Classifier-free guidance scale.",
                    },
                ),
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {"tooltip": "Sampling algorithm."},
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {"tooltip": "Sigma scheduler."},
                ),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Lower values keep more of the input latent.",
                    },
                ),
                "add_noise": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Generate noise from the selected Oneko profile. Disable this to sample with zero noise.",
                    },
                ),
                "noise_device": (
                    ["auto", "cpu", "cuda"],
                    {
                        "default": "auto",
                        "tooltip": "auto uses CPU for ComfyUI-core-like reproducibility. cuda falls back to CPU when unavailable.",
                    },
                ),
                "noise_profile": (
                    UNIVERSAL_NOISE_PROFILES,
                    {
                        "default": "gaussian",
                        "tooltip": "Basic noise type or composite Oneko profile.",
                    },
                ),
                "noise_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": "Final noise multiplier. For gaussian + auto, 1.0 keeps ComfyUI-core-like behavior.",
                    },
                ),
                "detail_bias": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Only affects composite profiles. Lower values emphasize larger forms; higher values emphasize details.",
                    },
                ),
                "preview_method": (
                    PREVIEW_METHODS,
                    {
                        "default": "default",
                        "tooltip": "Per-node latent preview override. For low memory, latent2rgb gives an early lightweight preview; none disables previews.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "INT")
    RETURN_NAMES = ("output", "denoised_output", "seed")
    FUNCTION = "sample"
    CATEGORY = "Oneko/05 Sampling"
    DESCRIPTION = "KSampler-style Universal Noise Sampler that builds the guider, sampler, and scheduler internally."

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        add_noise,
        noise_device,
        noise_profile="gaussian",
        noise_strength=1.0,
        detail_bias=0.35,
        preview_method="default",
    ):
        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)
        sampler = comfy.samplers.sampler_object(sampler_name)
        sigmas = _calculate_sigmas(model, scheduler, steps, denoise)
        if add_noise:
            noise = make_noise_generator(seed, noise_device, noise_profile, noise_strength, detail_bias)
        else:
            noise = OnekoEmptyNoise()

        return sample_custom_advanced(
            guider,
            sampler,
            sigmas,
            latent_image,
            noise,
            seed,
            preview_method=preview_method,
        )


def _calculate_sigmas(model, scheduler, steps, denoise):
    total_steps = int(steps)
    if denoise < 1.0:
        if denoise <= 0.0:
            return torch.FloatTensor([])
        total_steps = int(steps / denoise)

    sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, total_steps).cpu()
    return sigmas[-(int(steps) + 1) :]


NODE_CLASS_MAPPINGS = {
    "OnekoUniversalNoiseSampler": OnekoUniversalNoiseSampler,
    "OnekoUniversalNoiseSamplerAdvanced": OnekoUniversalNoiseSamplerAdvanced,
    "OnekoUniversalKSampler": OnekoUniversalKSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnekoUniversalNoiseSampler": "Universal Noise Sampler (Oneko)",
    "OnekoUniversalNoiseSamplerAdvanced": "Universal Noise Sampler Advanced (Oneko)",
    "OnekoUniversalKSampler": "Universal KSampler (Oneko)"
}
