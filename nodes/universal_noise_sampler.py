import torch

import comfy.samplers

from .noise_sampler_core import (
    MAX_SEED,
    PREVIEW_METHODS,
    UNIVERSAL_NOISE_PROFILES,
    OnekoEmptyNoise,
    guider_with_late_detail,
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
                        "tooltip": "Controls composite profiles and gaussian_detail (0 = Gaussian; higher = more spatial high-frequency noise). Has no effect on basic profiles such as pyramid_mix or violet.",
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
                        "tooltip": "gaussian is the native baseline. gaussian_detail gently raises spatial high frequencies using the same Gaussian seed. Other profiles can strongly change composition.",
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
                        "tooltip": "gaussian_detail: 0 is exactly Gaussian; higher values raise spatial high frequencies, keeping noise strength. Also controls composite profiles. No effect on basic profiles such as pyramid_mix or violet.",
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
            "optional": {
                "profile_blend": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "0 = native Gaussian for this seed, 1 = selected profile unchanged (legacy). Intermediate values mix the profile into Gaussian while preserving each sample's noise scale. Try 0.10-0.25 for strong profiles such as pyramid_mix or violet. This is not denoise strength.",
                    },
                ),
                "detail_boost": (
                    "FLOAT",
                    {
                        "default": 0.10,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.01,
                        "tooltip": "Optional late-sampling boost of fine spatial structure in the denoised prediction. Low spatial frequencies are excluded from the correction. New nodes start at 0.10; 0 disables the boost. Stronger values can produce harsh textures. No extra model evaluations. Spatial image/video predictions only.",
                    },
                ),
                "detail_start": (
                    "FLOAT",
                    {
                        "default": 0.65,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Start the detail boost this far into the incoming full sigma schedule. 0.65 leaves the first 65% of steps untouched; the effect fades in and out over the remaining sigma interval. Later starts reduce composition changes. 1 disables the boost.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "INT")
    RETURN_NAMES = ("output", "denoised_output", "seed")
    FUNCTION = "sample"
    CATEGORY = "Oneko/05 Sampling"
    DESCRIPTION = "Universal Noise Sampler with step ranges, Gaussian/profile blending and optional late spatial detail. Detail boost preserves early sampling and excludes coarse frequencies from its correction; image quality and composition still depend on the model and settings."

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
        profile_blend=1.0,
        detail_boost=0.0,
        detail_start=0.65,
    ):
        if add_noise:
            noise = make_noise_generator(noise_seed, noise_device, noise_profile, noise_strength, detail_bias, profile_blend)
        else:
            noise = OnekoEmptyNoise()

        guider = guider_with_late_detail(guider, sigmas, detail_boost, detail_start)
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
                        "tooltip": "Controls composite profiles and gaussian_detail (0 = Gaussian; higher = more spatial high-frequency noise). Has no effect on basic profiles such as pyramid_mix or violet.",
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
        sigmas = _calculate_sigmas(model, scheduler, steps, denoise, sampler_name)
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


def _calculate_sigmas(model, scheduler, steps, denoise, sampler_name="euler"):
    # Core owns sampler-specific sigma omissions (DPM2) and denoise slicing.
    schedule = comfy.samplers.KSampler(
        model, steps=int(steps), device="cpu", sampler=sampler_name,
        scheduler=scheduler, denoise=denoise,
    )
    return schedule.sigmas.cpu()


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
