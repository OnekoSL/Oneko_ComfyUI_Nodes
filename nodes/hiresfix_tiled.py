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
                        "default": True,
                        "tooltip": "Encode the upscaled image and attach it to positive conditioning as a reference latent.",
                    },
                ),
                "use_differential_diffusion": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Patch the model with Differential Diffusion before tiled refinement.",
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
                        "tooltip": "Initial tile noise distribution. gaussian with auto device and strength 1.0 matches ComfyUI core noise.",
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
                        "tooltip": "Universal tile noise profile. Leave as gaussian to preserve legacy noise_type behavior.",
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
        upscaled_image = _first_output(ImageUpscaleWithModel.execute(upscale_model, image))

        work_model = model
        if use_differential_diffusion:
            work_model = _first_output(DifferentialDiffusion.execute(work_model, differential_strength))

        work_positive = positive
        if use_reference_latent:
            latent = _first_output(VAEEncode().encode(vae, upscaled_image))
            work_positive = _first_output(ReferenceLatent.execute(work_positive, latent))

        usdu_class = _load_ultimate_sd_upscale_no_upscale()
        usdu_node = usdu_class()
        processing_globals = _get_usdu_processing_globals(usdu_class)
        original_sample = processing_globals["sample"]
        resolved_noise_profile = _resolve_noise_profile(noise_type, noise_profile)
        processing_globals["sample"] = _make_usdu_noise_sample(
            noise_device,
            resolved_noise_profile,
            noise_strength,
            detail_bias,
        )
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

        return (final_image, upscaled_image, seed)


def _first_output(value):
    if hasattr(value, "args"):
        return value.args[0]
    return value[0]


def _resolve_noise_profile(noise_type, noise_profile):
    if noise_profile == "gaussian" and noise_type != "gaussian":
        return noise_type
    return noise_profile


def _make_usdu_noise_sample(noise_device, noise_profile, noise_strength, detail_bias):
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
