import logging
import math

import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
import nodes
from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion
from comfy_extras.nodes_edit_model import ReferenceLatent
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
from nodes import VAEEncode

from .noise_sampler_core import NOISE_TYPES, UNIVERSAL_NOISE_PROFILES, make_noise_generator


def _combo_default(options, preferred):
    return preferred if preferred in options else options[0]


class OnekoHiResFixTiled:
    @classmethod
    def INPUT_TYPES(cls):
        samplers = comfy.samplers.KSampler.SAMPLERS
        schedulers = comfy.samplers.KSampler.SCHEDULERS
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_model": ("UPSCALE_MODEL",),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xffffffffffffffff,
                        "control_after_generate": True,
                        "tooltip": "Seed used by the tiled img2img refinement.",
                    },
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1}),
                "cfg": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (
                    samplers,
                    {"default": _combo_default(samplers, "res_2s")},
                ),
                "scheduler": (
                    schedulers,
                    {"default": _combo_default(schedulers, "beta57")},
                ),
                "denoise": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tile_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "tile_height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "mask_blur": ("INT", {"default": 64, "min": 0, "max": 64, "step": 1}),
                "tile_padding": ("INT", {"default": 192, "min": 0, "max": 8192, "step": 8}),
                "use_reference_latent": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Experimental extra reference conditioning, not ordinary img2img image preservation. Requires a checkpoint trained for compatible references. With factor-16 VAEs, uses the full upscaled image; smaller tiles do not reduce its reference memory.",
                    },
                ),
                "use_differential_diffusion": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Patch the model with Differential Diffusion. Progressive masking requires a sampler noise_mask, which USDU 1.7.2 does not supply here.",
                    },
                ),
                "differential_strength": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "tiled_decode": ("BOOLEAN", {"default": True}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1}),
                "noise_device": (
                    ["auto", "cpu", "cuda"],
                    {
                        "default": "auto",
                        "tooltip": "auto uses CPU for ComfyUI-core-like reproducibility. cuda falls back to CPU when unavailable.",
                    },
                ),
                "noise_type": (
                    NOISE_TYPES,
                    {
                        "default": "gaussian",
                        "tooltip": "Used only when noise_profile is gaussian. Set both noise_type and noise_profile to gaussian, with auto device and strength 1.0, for ComfyUI core noise.",
                    },
                ),
                "noise_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": "Multiplier applied to generated tile noise.",
                    },
                ),
                "noise_profile": (
                    UNIVERSAL_NOISE_PROFILES,
                    {
                        "default": "gaussian",
                        "tooltip": "A non-gaussian profile overrides noise_type. gaussian delegates to noise_type; set both to gaussian for ordinary Gaussian noise.",
                    },
                ),
                "detail_bias": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Only affects composite tile noise profiles such as illustrious_* and pony_v7_*.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT")
    RETURN_NAMES = ("image", "upscaled_image", "seed")
    FUNCTION = "hiresfix"
    CATEGORY = "Oneko/06 Image/Upscale"
    DESCRIPTION = "Model upscale plus optional ReferenceLatent/DifferentialDiffusion and Ultimate SD Upscale tiled refinement."

    def hiresfix(
        self,
        image,
        upscale_model,
        model,
        positive,
        negative,
        vae,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        tile_width,
        tile_height,
        mask_blur,
        tile_padding,
        use_reference_latent,
        use_differential_diffusion,
        differential_strength,
        tiled_decode,
        batch_size,
        noise_device="auto",
        noise_type="gaussian",
        noise_strength=1.0,
        noise_profile="gaussian",
        detail_bias=0.35,
    ):
        compression = int(vae.spacial_compression_encode())
        if compression == 16 and any(
            metadata.get("reference_latents")
            for conditioning in (positive, negative) for _, metadata in conditioning
        ):
            raise ValueError(
                "HiResFix Tiled cannot safely crop external reference_latents with a factor-16 VAE "
                "through Ultimate SD Upscale. Remove external references for ordinary img2img. "
                "Use this node's full-image reference option only with a reference-compatible checkpoint."
            )
        alignment = math.lcm(8, compression)
        tile_width = math.ceil((tile_width + tile_padding) / alignment) * alignment - tile_padding
        tile_height = math.ceil((tile_height + tile_padding) / alignment) * alignment - tile_padding
        upscaled_image = _first_output(ImageUpscaleWithModel.execute(upscale_model, image))

        work_model = model
        if use_differential_diffusion:
            work_model = _first_output(DifferentialDiffusion.execute(work_model, differential_strength))

        work_positive = positive
        full_image_reference = None
        if use_reference_latent:
            latent = _first_output(VAEEncode().encode(vae, upscaled_image))
            if compression == 16:
                # USDU crops references with a fixed factor 8 and only the first
                # tile region. Attach our full-image reference after that crop.
                full_image_reference = latent
            else:
                work_positive = _first_output(ReferenceLatent.execute(work_positive, latent))

        usdu_class = _load_ultimate_sd_upscale_no_upscale()
        usdu_node = usdu_class()
        processing_globals = _get_usdu_processing_globals(usdu_class)
        original_sample = processing_globals["sample"]
        resolved_noise_profile = _resolve_noise_profile(noise_type, noise_profile)
        tile_sample = _make_usdu_noise_sample(
            noise_device,
            resolved_noise_profile,
            noise_strength,
            detail_bias,
            reference_latent=full_image_reference,
            warn_missing_noise_mask=use_differential_diffusion,
        )
        processing_globals["sample"] = tile_sample
        try:
            final_image = _first_output(
                usdu_node.upscale(
                    upscaled_image,
                    work_model,
                    work_positive,
                    negative,
                    vae,
                    seed,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    denoise,
                    "Linear",
                    tile_width,
                    tile_height,
                    mask_blur,
                    tile_padding,
                    "None",
                    1.0,
                    8,
                    64,
                    16,
                    True,
                    tiled_decode,
                    batch_size,
                )
            )
        finally:
            processing_globals["sample"] = original_sample
            if tile_sample.missing_noise_mask:
                # USDU suppresses root logging during its tile loop. Report only
                # after it restores logging, once for this whole invocation.
                logging.warning(
                    "Oneko HiResFix: Ultimate SD Upscale supplied no noise_mask; "
                    "Differential Diffusion cannot apply progressive masking to these tiles. "
                    "USDU still composites the decoded tiles with its pixel-space masks."
                )

        return (final_image, upscaled_image, seed)


def _first_output(value):
    if hasattr(value, "args"):
        return value.args[0]
    return value[0]


def _resolve_noise_profile(noise_type, noise_profile):
    if noise_profile == "gaussian" and noise_type != "gaussian":
        return noise_type
    return noise_profile


def _make_usdu_noise_sample(noise_device, noise_profile, noise_strength, detail_bias,
                            reference_latent=None, warn_missing_noise_mask=False):
    def sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise, custom_sampler, custom_sigmas):
        latent_samples = latent["samples"]
        fixed_samples = comfy.sample.fix_empty_latent_channels(
            model,
            latent_samples,
            latent.get("downscale_ratio_spacial", None),
        )
        work_latent = latent.copy()
        work_latent["samples"] = fixed_samples

        noise = make_noise_generator(
            seed,
            noise_device,
            noise_profile,
            noise_strength,
            detail_bias,
        ).generate_noise(work_latent)

        noise_mask = None
        if "noise_mask" in work_latent:
            noise_mask = work_latent["noise_mask"]

        if noise_mask is None and warn_missing_noise_mask:
            sample.missing_noise_mask = True

        if reference_latent is not None:
            reference = reference_latent["samples"]
            # USDU batches image-major: all tile regions of image 0, then image 1.
            tile_count, remainder = divmod(fixed_samples.shape[0], reference.shape[0])
            if remainder or tile_count < 1:
                raise ValueError("HiResFix tile batch does not match the full-image reference batch.")
            if tile_count > 1:
                reference = reference.repeat_interleave(tile_count, dim=0)
            positive = _first_output(ReferenceLatent.execute(positive, {"samples": reference}))

        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = comfy.sample.sample(
            model,
            noise,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            fixed_samples,
            denoise=denoise,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )

        out = work_latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out["samples"] = samples
        return out

    sample.missing_noise_mask = False
    return sample


def _get_usdu_processing_globals(usdu_class):
    for cls in usdu_class.mro():
        upscale = cls.__dict__.get("upscale")
        if upscale is None:
            continue
        stable_diffusion_processing = upscale.__globals__.get("StableDiffusionProcessing")
        if stable_diffusion_processing is not None:
            processing_globals = stable_diffusion_processing.__init__.__globals__
            if "sample" in processing_globals:
                return processing_globals
    raise RuntimeError("Could not locate UltimateSDUpscale processing sample function.")


def _load_ultimate_sd_upscale_no_upscale():
    try:
        return nodes.NODE_CLASS_MAPPINGS["UltimateSDUpscaleNoUpscale"]
    except KeyError:
        raise ImportError(
            "UltimateSDUpscaleNoUpscale is not loaded. Install or enable "
            "ComfyUI_UltimateSDUpscale, then restart ComfyUI."
        ) from None


NODE_CLASS_MAPPINGS = {
    "OnekoHiResFixTiled": OnekoHiResFixTiled,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnekoHiResFixTiled": "HiResFix Tiled (Oneko)"
}
