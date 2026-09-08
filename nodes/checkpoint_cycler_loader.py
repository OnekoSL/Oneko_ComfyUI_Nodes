import os

import comfy.sd
import folder_paths


def _model_name(ckpt_name):
    normalized = ckpt_name.replace("\\", "/")
    filename = normalized.rsplit("/", 1)[-1]
    return os.path.splitext(filename)[0]


def _folder_name(ckpt_name):
    normalized = ckpt_name.replace("\\", "/")
    if "/" not in normalized:
        return ""
    return normalized.rsplit("/", 1)[0]


def _load_checkpoint(ckpt_name):
    ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
    out = comfy.sd.load_checkpoint_guess_config(
        ckpt_path,
        output_vae=True,
        output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )
    return out[:3]


class OnekoCheckpointCyclerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (
                    folder_paths.get_filename_list("checkpoints"),
                    {
                        "control_after_generate": True,
                        "tooltip": "Checkpoint to load. The frontend can increment, decrement, randomize, or wrap this combo after queueing.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "modelname", "ckpt_name", "folder")
    FUNCTION = "load_checkpoint"
    CATEGORY = "Oneko/01 Loaders"
    DESCRIPTION = "Loads a checkpoint and exposes model name metadata, with combo control-after-generate support."

    def load_checkpoint(self, ckpt_name):
        model, clip, vae = _load_checkpoint(ckpt_name)
        return (model, clip, vae, _model_name(ckpt_name), ckpt_name, _folder_name(ckpt_name))

    @classmethod
    def VALIDATE_INPUTS(cls, ckpt_name):
        if folder_paths.get_full_path("checkpoints", ckpt_name) is None:
            return "Invalid checkpoint file: {}".format(ckpt_name)
        return True


NODE_CLASS_MAPPINGS = {
    "OnekoCheckpointCyclerLoader": OnekoCheckpointCyclerLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnekoCheckpointCyclerLoader": "Checkpoint Cycler Loader (Oneko)"
}
