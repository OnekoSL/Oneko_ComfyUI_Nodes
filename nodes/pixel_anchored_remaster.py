import comfy.samplers
import comfy.utils
from comfy_extras.nodes_edit_model import ReferenceLatent
from nodes import LatentUpscale, VAEDecodeTiled, VAEEncodeTiled, common_ksampler


PIXEL_METHODS = ["area", "bicubic", "nearest-exact", "bilinear", "lanczos"]


def _combo_default(options, preferred):
    return preferred if preferred in options else options[0]


def _multiple_of_8(value):
    return max(8, int(round(value)) // 8 * 8)


def _scaled_size(width, height, scale):
    scale = min(1.0, max(0.25, float(scale)))
    return (_multiple_of_8(width * scale), _multiple_of_8(height * scale))


def _resize_image(image, width, height, method):
    samples = image.movedim(-1, 1)
    resized = comfy.utils.common_upscale(samples, width, height, method, "disabled")
    return resized.movedim(1, -1)


def _first_output(value):
    if hasattr(value, "args"):
        return value.args[0]
    return value[0]


class OnekoPixelAnchoredRemaster:
    @classmethod
    def INPUT_TYPES(cls):
        samplers = comfy.samplers.KSampler.SAMPLERS
        schedulers = comfy.samplers.KSampler.SCHEDULERS
        return {
            "required": {
                "image": ("IMAGE",),
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
                        "tooltip": "Seed used by the low-denoise remaster pass.",
                    },
                ),
                "pixel_scale": (
                    "FLOAT",
                    {
                        "default": 0.90,
                        "min": 0.25,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Pixel-space downscale before latent remastering. Lower values calm artifacts more but drift more.",
                    },
                ),
                "pixel_method": (PIXEL_METHODS, {"default": "area"}),
                "latent_upscale_method": (
                    LatentUpscale.upscale_methods,
                    {"default": _combo_default(LatentUpscale.upscale_methods, "bislerp")},
                ),
                "steps": ("INT", {"default": 12, "min": 1, "max": 10000, "step": 1}),
                "cfg": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (samplers, {"default": _combo_default(samplers, "res_2m")}),
                "scheduler": (schedulers, {"default": _combo_default(schedulers, "beta57")}),
                "denoise": ("FLOAT", {"default": 0.16, "min": 0.0, "max": 1.0, "step": 0.01}),
                "remaster_blend": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Amount of remastered image blended over the original pixel anchor. Lower values preserve the input more.",
                    },
                ),
                "vae_tile_size": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 32}),
                "vae_overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                "use_reference_latent": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Attach the downscaled encoded image as a reference latent for identity/detail stability.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "LATENT", "INT", "STRING")
    RETURN_NAMES = ("final_image", "downscaled_image", "remaster_latent", "seed", "settings_report")
    FUNCTION = "remaster"
    CATEGORY = "Oneko/06 Image/Upscale"
    DESCRIPTION = "Anima-safe pixel downscale, tiled VAE encode, latent upscale, and low-denoise remaster pass."

    def remaster(
        self,
        image,
        model,
        positive,
        negative,
        vae,
        seed,
        pixel_scale,
        pixel_method,
        latent_upscale_method,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        remaster_blend,
        vae_tile_size,
        vae_overlap,
        use_reference_latent,
    ):
        _batch, input_h, input_w, _channels = image.shape
        final_w = _multiple_of_8(input_w)
        final_h = _multiple_of_8(input_h)
        down_w, down_h = _scaled_size(input_w, input_h, pixel_scale)

        downscaled_image = _resize_image(image, down_w, down_h, pixel_method)
        encoded_latent = _first_output(VAEEncodeTiled().encode(vae, downscaled_image, vae_tile_size, vae_overlap))
        remaster_latent = _first_output(
            LatentUpscale().upscale(encoded_latent, latent_upscale_method, final_w, final_h, "disabled")
        )

        work_positive = positive
        if use_reference_latent:
            work_positive = _first_output(ReferenceLatent.execute(work_positive, encoded_latent))

        sampled = False
        if float(denoise) > 0.0:
            remaster_latent = _first_output(
                common_ksampler(
                    model,
                    seed,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    work_positive,
                    negative,
                    remaster_latent,
                    denoise=denoise,
                )
            )
            sampled = True

        raw_remaster_image = _first_output(VAEDecodeTiled().decode(vae, remaster_latent, vae_tile_size, vae_overlap))
        final_image = _blend_with_anchor(image, raw_remaster_image, remaster_blend)
        report = _settings_report(
            input_w,
            input_h,
            down_w,
            down_h,
            final_w,
            final_h,
            remaster_latent,
            seed,
            sampled,
            pixel_scale,
            pixel_method,
            latent_upscale_method,
            steps,
            cfg,
            sampler_name,
            scheduler,
            denoise,
            remaster_blend,
            vae_tile_size,
            vae_overlap,
            use_reference_latent,
        )
        return (final_image, downscaled_image, remaster_latent, seed, report)


def _settings_report(
    input_w,
    input_h,
    down_w,
    down_h,
    final_w,
    final_h,
    remaster_latent,
    seed,
    sampled,
    pixel_scale,
    pixel_method,
    latent_upscale_method,
    steps,
    cfg,
    sampler_name,
    scheduler,
    denoise,
    remaster_blend,
    vae_tile_size,
    vae_overlap,
    use_reference_latent,
):
    latent_shape = tuple(remaster_latent["samples"].shape)
    lines = [
        "Pixel Anchored Remaster (Oneko)",
        f"seed: {seed}",
        f"sampled: {sampled}",
        f"input_size: {input_w}x{input_h}",
        f"downscaled_size: {down_w}x{down_h}",
        f"final_size: {final_w}x{final_h}",
        f"remaster_latent_shape: {latent_shape}",
        f"pixel_scale: {pixel_scale}",
        f"pixel_method: {pixel_method}",
        f"latent_upscale_method: {latent_upscale_method}",
        f"steps: {steps}",
        f"cfg: {cfg}",
        f"sampler_name: {sampler_name}",
        f"scheduler: {scheduler}",
        f"denoise: {denoise}",
        f"remaster_blend: {remaster_blend}",
        f"vae_tile_size: {vae_tile_size}",
        f"vae_overlap: {vae_overlap}",
        f"use_reference_latent: {use_reference_latent}",
    ]
    return "\n".join(lines)


def _blend_with_anchor(anchor_image, remaster_image, remaster_blend):
    blend = min(1.0, max(0.0, float(remaster_blend)))
    if blend >= 1.0:
        return remaster_image
    if blend <= 0.0:
        target_h = remaster_image.shape[1]
        target_w = remaster_image.shape[2]
        return _resize_image(anchor_image, target_w, target_h, "lanczos")

    anchor = anchor_image
    if anchor.shape[1] != remaster_image.shape[1] or anchor.shape[2] != remaster_image.shape[2]:
        target_h = remaster_image.shape[1]
        target_w = remaster_image.shape[2]
        anchor = _resize_image(anchor, target_w, target_h, "lanczos")

    return (anchor * (1.0 - blend) + remaster_image * blend).clamp(0.0, 1.0)


NODE_CLASS_MAPPINGS = {
    "OnekoPixelAnchoredRemaster": OnekoPixelAnchoredRemaster,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnekoPixelAnchoredRemaster": "Pixel Anchored Remaster (Oneko)"
}
