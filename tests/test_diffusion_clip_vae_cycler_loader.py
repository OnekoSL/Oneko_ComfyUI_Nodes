from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from Oneko_ComfyUI_Nodes.nodes import diffusion_clip_vae_cycler_loader as loader


UNET_NAME = "MOSS/Fwaun test.safetensors"
CLIP_NAME = "MOSS/qwen3.5_9b_w4a8_gptq.safetensors"
VAE_NAME = "MOSS/flux2-vae_small.safetensors"


@pytest.fixture
def model_inventory(monkeypatch):
    names = {
        "diffusion_models": [UNET_NAME],
        "text_encoders": [CLIP_NAME],
    }
    monkeypatch.setattr(loader.folder_paths, "get_filename_list", lambda category: names.get(category, []))
    monkeypatch.setattr(
        loader.folder_paths,
        "get_full_path",
        lambda category, name: "/models/" + name if name in names.get(category, []) else None,
    )
    monkeypatch.setattr(loader, "_vae_names", lambda: [VAE_NAME])


def test_clip_menu_and_validation_accept_native_core_types(model_inventory):
    native_types = tuple(loader.CLIPLoader.INPUT_TYPES()["required"]["type"][0])
    schema = loader.OnekoDiffusionClipVaeCyclerLoader.INPUT_TYPES()["required"]
    assert tuple(schema["clip_type"][0]) == native_types
    assert "moss_image" in native_types
    assert schema["clip_type"][1]["default"] == "stable_diffusion"
    assert schema["unet_name"][1]["control_after_generate"] is True
    for clip_type in native_types:
        assert loader.OnekoDiffusionClipVaeCyclerLoader.VALIDATE_INPUTS(
            UNET_NAME, CLIP_NAME, VAE_NAME, clip_type=clip_type
        ) is True


def test_core_type_changes_are_shared_by_menu_and_validation(monkeypatch, model_inventory):
    native_types = ["stable_diffusion", "moss_image", "future_native_type"]
    monkeypatch.setattr(
        loader.CLIPLoader,
        "INPUT_TYPES",
        classmethod(lambda cls: {"required": {"type": (native_types,)}}),
    )
    menu = loader.OnekoDiffusionClipVaeCyclerLoader.INPUT_TYPES()["required"]["clip_type"][0]
    assert tuple(menu) == tuple(native_types)
    assert loader.OnekoDiffusionClipVaeCyclerLoader.VALIDATE_INPUTS(
        UNET_NAME, CLIP_NAME, VAE_NAME, clip_type="future_native_type"
    ) is True
    assert loader.OnekoDiffusionClipVaeCyclerLoader.VALIDATE_INPUTS(
        UNET_NAME, CLIP_NAME, VAE_NAME, clip_type="not_a_native_type"
    ) == "Invalid CLIP type: not_a_native_type"


@pytest.mark.parametrize(
    ("is_krea2", "requested_type", "expected_type"),
    [
        (False, "moss_image", "moss_image"),
        (False, "stable_diffusion", "stable_diffusion"),
        (False, "flux2", "flux2"),
        (False, "krea2", "krea2"),
        (True, "stable_diffusion", "krea2"),
        (True, "moss_image", "moss_image"),
        (True, "flux2", "flux2"),
        (True, "krea2", "krea2"),
    ],
)
def test_native_loading_preserves_explicit_types_krea2_and_outputs(
    monkeypatch, is_krea2, requested_type, expected_type
):
    class Krea2:
        pass

    monkeypatch.setattr(loader.comfy.model_base, "Krea2", Krea2)
    model = SimpleNamespace(model=Krea2() if is_krea2 else object())
    clip, vae = object(), object()
    load_unet = Mock(return_value=(model,))
    load_clip = Mock(return_value=(clip,))
    load_vae = Mock(return_value=(vae,))
    monkeypatch.setattr(loader.UNETLoader, "load_unet", load_unet)
    monkeypatch.setattr(loader.CLIPLoader, "load_clip", load_clip)
    monkeypatch.setattr(loader.VAELoader, "load_vae", load_vae)

    result = loader.OnekoDiffusionClipVaeCyclerLoader().load_models(
        UNET_NAME,
        CLIP_NAME,
        VAE_NAME,
        weight_dtype="fp8_e4m3fn",
        clip_type=requested_type,
        clip_device="cpu",
    )

    load_unet.assert_called_once_with(UNET_NAME, "fp8_e4m3fn")
    load_clip.assert_called_once_with(CLIP_NAME, expected_type, "cpu")
    load_vae.assert_called_once_with(VAE_NAME)
    assert result == (
        model, clip, vae, "Fwaun test", "Fwaun_test", UNET_NAME, CLIP_NAME, VAE_NAME, "MOSS"
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"unet_name": "missing.safetensors"}, "Invalid diffusion model file: missing.safetensors"),
        ({"clip_name": "missing.safetensors"}, "Invalid CLIP/text encoder file: missing.safetensors"),
        ({"vae_name": "missing.safetensors"}, "Invalid VAE file: missing.safetensors"),
        ({"weight_dtype": "unknown"}, "Invalid weight dtype: unknown"),
        ({"clip_type": "unknown"}, "Invalid CLIP type: unknown"),
        ({"clip_device": "cuda"}, "Invalid CLIP device: cuda"),
    ],
)
def test_loader_still_rejects_invalid_inputs(model_inventory, changes, message):
    inputs = dict(unet_name=UNET_NAME, clip_name=CLIP_NAME, vae_name=VAE_NAME, clip_type="moss_image")
    inputs.update(changes)
    assert loader.OnekoDiffusionClipVaeCyclerLoader.VALIDATE_INPUTS(**inputs) == message
