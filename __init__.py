"""Curated Oneko nodes for ComfyUI."""

from .nodes import (
    ace_song_timeline_conditioning,
    ace_song_variation_director,
    audio_timeline_mixer,
    checkpoint_cycler_loader,
    clip_sculpt_text_encode,
    conditioning_tools,
    diffusion_clip_vae_cycler_loader,
    hiresfix_tiled,
    incrementing_int_string,
    load_image_with_subfolders,
    minimax_h3_prompt_builder,
    noise_profile_cycler,
    ollama_prompt_refiner,
    ollama_video_prompt_refiner,
    ollama_vision_captioner,
    pixel_anchored_remaster,
    random_vocab_string_list,
    regional_prompt_encoder,
    regional_sculpt_prompt_encoder,
    regional_split_regions,
    t5_equal_length_balancer,
    t5_sculpt_equal_length_balancer,
    unet_block_noise_patch,
    universal_noise_sampler,
    wan22_video_toolkit,
)

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for module in (
    ace_song_timeline_conditioning,
    ace_song_variation_director,
    audio_timeline_mixer,
    checkpoint_cycler_loader,
    clip_sculpt_text_encode,
    conditioning_tools,
    diffusion_clip_vae_cycler_loader,
    hiresfix_tiled,
    incrementing_int_string,
    load_image_with_subfolders,
    minimax_h3_prompt_builder,
    noise_profile_cycler,
    ollama_prompt_refiner,
    ollama_video_prompt_refiner,
    ollama_vision_captioner,
    pixel_anchored_remaster,
    random_vocab_string_list,
    regional_prompt_encoder,
    regional_sculpt_prompt_encoder,
    regional_split_regions,
    t5_equal_length_balancer,
    t5_sculpt_equal_length_balancer,
    unet_block_noise_patch,
    universal_noise_sampler,
    wan22_video_toolkit,
):
    duplicate_ids = NODE_CLASS_MAPPINGS.keys() & module.NODE_CLASS_MAPPINGS.keys()
    if duplicate_ids:
        raise RuntimeError(f"Duplicate Oneko node IDs: {sorted(duplicate_ids)}")
    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
