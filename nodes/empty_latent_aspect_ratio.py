import math

import nodes as core_nodes
from comfy_api.latest import io
from comfy_extras.nodes_flux import EmptyFlux2LatentImage
from comfy_extras.nodes_resolution import AspectRatio, ResolutionSelector
from comfy_extras.nodes_sd3 import EmptySD3LatentImage


EXTRA_ASPECT_RATIOS = {
    "5:4 (Classic Monitor)": (5, 4),
    "16:10 (Desktop Wallpaper)": (16, 10),
    "32:9 (Super Ultrawide)": (32, 9),
    "32:10 (Dual Monitor)": (32, 10),
    "9:20 (Phone Wallpaper)": (9, 20),
    "2:1 (Banner)": (2, 1),
    "3:1 (Wide Banner)": (3, 1),
    "4:1 (Panoramic Banner)": (4, 1),
    "5:1 (Header Banner)": (5, 1),
}
ASPECT_RATIOS = ("custom", *(ratio.value for ratio in AspectRatio), *EXTRA_ASPECT_RATIOS)
LATENT_FORMATS = ("auto", "SD1 / SDXL", "SD3 / FLUX.1", "MOSS / FLUX.2")


class OnekoEmptyLatentAspectRatio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OnekoEmptyLatentAspectRatio",
            display_name="Empty Latent Aspect Ratio (Oneko)",
            category="Oneko/05 Sampling/Latent",
            description="Create empty image latents from an aspect ratio or custom dimensions, with scale metadata for correct model-specific sampling sizes.",
            search_aliases=["empty latent", "aspect ratio", "banner", "wallpaper", "MOSS", "FLUX.2"],
            inputs=[
                io.Combo.Input("aspect_ratio", options=list(ASPECT_RATIOS), default=AspectRatio.PHOTO_V.value,
                               tooltip="Photo, wallpaper or banner aspect ratio; custom uses width and height. Use swap_dimensions to rotate any preset."),
                io.Float.Input("megapixels", default=1.5, min=0.1, max=16.0, step=0.1,
                               tooltip="Preset resolution budget, as in Core Resolution Selector: 1 MP = 1024 x 1024 pixels. Ignored for custom dimensions."),
                io.Int.Input("width", default=1024, min=16, max=core_nodes.MAX_RESOLUTION, step=8,
                             tooltip="Custom width before swapping and scaling; used only when aspect_ratio is custom."),
                io.Int.Input("height", default=1536, min=16, max=core_nodes.MAX_RESOLUTION, step=8,
                             tooltip="Custom height before swapping and scaling; used only when aspect_ratio is custom."),
                io.Boolean.Input("swap_dimensions", default=False),
                io.Float.Input("scale_factor", default=1.0, min=0.125, max=8.0, step=0.05,
                               tooltip="Scale both sides after selecting the resolution. Output dimensions are rounded to the latent format's grid."),
                io.Combo.Input("latent_format", options=list(LATENT_FORMATS), default="auto",
                               tooltip="auto lets the sampler adapt the empty latent to its model. MOSS / FLUX.2 creates 128 channels at 1/16 resolution directly."),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
            ],
            outputs=[io.Latent.Output("latent"), io.Int.Output("width"), io.Int.Output("height")],
        )

    @classmethod
    def execute(cls, aspect_ratio=AspectRatio.PHOTO_V.value, megapixels=1.5, width=1024, height=1536,
                swap_dimensions=False, scale_factor=1.0, latent_format="auto", batch_size=1):
        if latent_format not in LATENT_FORMATS:
            raise ValueError(f"Unknown latent format: {latent_format}")
        alignment = 16 if latent_format in ("auto", "MOSS / FLUX.2") else 8
        if aspect_ratio in EXTRA_ASPECT_RATIOS:
            w_ratio, h_ratio = EXTRA_ASPECT_RATIOS[aspect_ratio]
            scale = math.sqrt(megapixels * 1024 * 1024 / (w_ratio * h_ratio))
            width = round(w_ratio * scale / alignment) * alignment
            height = round(h_ratio * scale / alignment) * alignment
        elif aspect_ratio != "custom":
            width, height = ResolutionSelector.execute(aspect_ratio, megapixels, alignment).result
        if swap_dimensions:
            width, height = height, width
        width = max(16, round(width * scale_factor / alignment) * alignment)
        height = max(16, round(height * scale_factor / alignment) * alignment)

        if latent_format == "MOSS / FLUX.2":
            latent = EmptyFlux2LatentImage.execute(width, height, batch_size).result[0]
            latent["downscale_ratio_spacial"] = 16
        elif latent_format == "SD3 / FLUX.1":
            latent = EmptySD3LatentImage.execute(width, height, batch_size).result[0]
        else:
            latent = core_nodes.EmptyLatentImage().generate(width, height, batch_size)[0]
        return io.NodeOutput(latent, width, height)


NODE_CLASS_MAPPINGS = {
    "OnekoEmptyLatentAspectRatio": OnekoEmptyLatentAspectRatio,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OnekoEmptyLatentAspectRatio": "Empty Latent Aspect Ratio (Oneko)",
}
