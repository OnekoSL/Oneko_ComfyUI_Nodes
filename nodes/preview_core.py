"""Per-callback preview choices using ComfyUI's native preview decoders."""

import logging

import comfy.utils
import folder_paths
import latent_preview
from comfy.cli_args import LatentPreviewMethod


def _get_previewer(device, latent_format, method):
    if method == LatentPreviewMethod.NoPreviews:
        return None
    if method == LatentPreviewMethod.TAESD:
        decoder_path = None
        if latent_format.taesd_decoder_name is not None:
            name = next((name for name in folder_paths.get_filename_list("vae_approx")
                         if name.startswith(latent_format.taesd_decoder_name)), "")
            decoder_path = folder_paths.get_full_path("vae_approx", name)
        if decoder_path:
            if latent_format.taesd_decoder_name in latent_preview.VIDEO_TAES:
                vae = latent_preview.VAE(comfy.utils.load_torch_file(decoder_path))
                vae.first_stage_model.show_progress_bar = False
                return latent_preview.TAEHVPreviewerImpl(vae, compile_preview=latent_format.compile_preview)
            taesd = latent_preview.TAESD(None, decoder_path, latent_channels=latent_format.latent_channels).to(device)
            return latent_preview.TAESDPreviewerImpl(taesd)
        logging.warning("Warning: TAESD previews enabled, but could not find models/vae_approx/%s",
                        latent_format.taesd_decoder_name)
    # Core Auto and missing TAESD decoders fall back to the latent RGB factors.
    if latent_format.latent_rgb_factors is not None:
        return latent_preview.Latent2RGBPreviewer(
            latent_format.latent_rgb_factors,
            latent_format.latent_rgb_factors_bias,
            latent_format.latent_rgb_factors_reshape,
        )
    return None


def prepare_preview_callback(model, steps, x0_output_dict=None, preview_method="default"):
    if preview_method in (None, "", "default"):
        return latent_preview.prepare_callback(model, steps, x0_output_dict)
    method = LatentPreviewMethod.from_string(preview_method)
    if method is None:
        # Match the former setter's invalid-value fallback without changing args.
        method = latent_preview.default_preview_method
    previewer = _get_previewer(model.load_device, model.model.latent_format, method)
    progress = comfy.utils.ProgressBar(steps)

    def callback(step, x0, x, total_steps):
        if x0_output_dict is not None:
            x0_output_dict["x0"] = x0
        preview_bytes = None
        if previewer is not None:
            preview_latent = x0.tensors[0] if x0.is_nested else x0
            preview_bytes = previewer.decode_latent_to_preview_image("JPEG", preview_latent)
        progress.update_absolute(step + 1, total_steps, preview_bytes)

    return callback
