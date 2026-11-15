import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import comfy.sampler_helpers
import comfy.samplers
from comfy.conds import CONDRegular
from comfy.ldm.moss_image import model as moss_core
from comfy.model_patcher import ModelPatcher, create_model_options_clone
from comfy.patcher_extension import WrappersMP
from comfy_extras.nodes_slg import SkipLayerGuidanceDiT, SkipLayerGuidanceDiTSimple
from Oneko_ComfyUI_Nodes.nodes import dit_block_noise_patch as dit_noise


class WidthZeros(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width

    def forward(self, x):
        return x.new_zeros((x.shape[0], self.width))


class ImageEmbedding(torch.nn.Module):
    def forward(self, x):
        return x[..., :16]


class ImageHead(torch.nn.Module):
    def forward(self, x):
        return x.repeat(1, 1, 8)


class TextEmbedding(torch.nn.Module):
    def forward(self, x, pe, transformer_options, mask):
        return x


class RecordingBlock(torch.nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
        self.calls = []

    def forward(self, x, mod, pe, transformer_options, ref=None, ref_kv=None, kv_out=None, mask=None):
        before = x.clone()
        x.add_((self.index + 1) * 0.125)  # native MOSS blocks also update their working sequence in place
        if kv_out is not None:
            kv_out[:, 0].copy_(x)
            kv_out[:, 1].copy_(x + 1)
        self.calls.append({"input": before, "output": x.clone(), "ref": ref,
                           "ref_kv": None if ref_kv is None else ref_kv.clone()})
        return x


class ModelHost(torch.nn.Module):
    def __init__(self, diffusion):
        super().__init__()
        self.diffusion_model = diffusion
        self.model_sampling = SimpleNamespace(percent_to_sigma=lambda value: 1.0 - value)
        self.current_patcher = None

    def memory_required(self, input_shape, cond_shapes):
        return 0

    def apply_model(self, x, timestep, c_crossattn, transformer_options, **kwargs):
        return self.diffusion_model(x, timestep, c_crossattn, transformer_options, **kwargs)


@pytest.fixture
def model_factory(monkeypatch):
    # Keep native forward, RoPE, token assembly, reference handling and block dispatch;
    # only the final fused normalization uses a small CPU-only mathematical stand-in.
    monkeypatch.setattr(moss_core, "modulated_rms_norm", lambda x, norm, scale, shift: norm(x) * (1 + scale) + shift)

    def build():
        model = moss_core.MOSSImage.__new__(moss_core.MOSSImage)
        torch.nn.Module.__init__(model)
        model.dtype = torch.float32
        model.latent_channels = 128
        model.rope_axes_dim = [4, 4, 4, 4]
        model.video_embedder = ImageEmbedding()
        model.text_embedder = TextEmbedding()
        model.time_embedding = WidthZeros(16)
        model.shared_block_modulation = WidthZeros(96)
        model.video_head_modulation = WidthZeros(32)
        model.layers = torch.nn.ModuleList([RecordingBlock(i) for i in range(3)])
        model.final_norm_video = torch.nn.Identity()
        model.final_linear_video = ImageHead()
        host = ModelHost(model)
        return ModelPatcher(host, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    return build


def noise_patch(model, **overrides):
    values = dict(blocks="all", noise_type="gaussian", noise_strength=0.2, seed=123,
                  start_percent=0.0, end_percent=1.0)
    values.update(overrides)
    return dit_noise.OnekoDiTBlockNoisePatch.execute(model, **values).result[0]


def runtime_options(patcher, sigma=0.5):
    options = create_model_options_clone(patcher.model_options)
    comfy.sampler_helpers.prepare_model_patcher(patcher, {}, options)
    options["transformer_options"]["sigmas"] = torch.tensor([sigma])
    return options


def run_model(patcher, height=3, width=5, text_length=5, sigma=0.5, dtype=torch.float32, **kwargs):
    x = torch.linspace(0.2, 1.2, 2 * 128 * height * width, dtype=dtype).reshape(2, 128, height, width)
    context = torch.arange(2 * text_length * 16, dtype=torch.float32).reshape(2, text_length, 16).to(dtype) + 10
    original_x, original_context = x.clone(), context.clone()
    options = runtime_options(patcher, sigma)
    before = copy.deepcopy(options["transformer_options"])
    output = patcher.get_model_object("diffusion_model")(x, torch.tensor([sigma]), context,
                                                         options["transformer_options"], **kwargs)
    torch.testing.assert_close(x, original_x, rtol=0, atol=0)
    torch.testing.assert_close(context, original_context, rtol=0, atol=0)
    assert options["transformer_options"].keys() == before.keys()
    assert options["transformer_options"].get("patches_replace") == before.get("patches_replace")
    return output


def observe_blocks(patcher):
    observed = {}
    for index in range(3):
        def observe(args, extra, index=index):
            result = extra["original_block"](args)
            observed[index] = result["img"].clone()
            return result
        patcher.set_model_patch_replace(observe, "dit", "single_block", index)
    return observed


@pytest.mark.parametrize("blocks,selected", [
    ("all", {0, 1, 2}), ("0", {0}), ("0, 2", {0, 2}),
    ("0-1", {0, 1}), ("0, 0, 2", {0, 2}), ("", set()), ("  ", set()),
])
def test_block_selection_changes_only_selected_original_outputs(model_factory, blocks, selected):
    original = model_factory()
    observed = observe_blocks(original)
    patched = noise_patch(original, blocks=blocks)
    run_model(patched)
    for index, block in enumerate(original.model.diffusion_model.layers):
        native_result = block.calls[-1]["output"]
        actual = observed[index]
        assert torch.equal(actual[:, 15:], native_result[:, 15:])
        assert (not torch.equal(actual[:, :15], native_result[:, :15])) == (index in selected)


def test_native_moss_noise_preserves_the_sequence_buffer_across_blocks(model_factory):
    # MOSS allocates its working sequence before the AIMDO memory compiler region.
    # Replacing that tensor inside a block leaks an allocation across block scopes.
    original = model_factory()
    buffers = []

    def observe(args, extra):
        incoming = args["img"]
        result = extra["original_block"](args)
        assert result["img"] is incoming
        buffers.append((id(incoming), incoming.data_ptr()))
        return result

    for index in range(3):
        original.set_model_patch_replace(observe, "dit", "single_block", index)

    with patch.object(dit_noise, "generate_oneko_noise_for_tensor", wraps=dit_noise.generate_oneko_noise_for_tensor) as generate:
        output = run_model(noise_patch(original, blocks="all"))

    assert generate.call_count == 3
    assert len(buffers) == 3 and len(set(buffers)) == 1
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("blocks", ["-1", "3", "0-3", "2-0", "x", "1.5", "0,,1"])
def test_invalid_block_selection_is_rejected_without_mutation(model_factory, blocks):
    original = model_factory()
    before = copy.deepcopy(original.model_options)
    with pytest.raises(ValueError):
        noise_patch(original, blocks=blocks)
    assert original.model_options == before
    assert original.wrappers == {}


def test_zero_strength_is_exactly_neutral_and_does_not_generate_noise(model_factory):
    original = model_factory()
    with patch.object(dit_noise, "generate_oneko_noise_for_tensor", side_effect=AssertionError("no noise expected")):
        patched = noise_patch(original, noise_strength=0)
        torch.testing.assert_close(run_model(patched), run_model(original), rtol=0, atol=0)
    assert original.wrappers == {}


@pytest.mark.parametrize("reference_mode", ["none", "index", "index_timestep_zero", "cached", "preprocess"])
@pytest.mark.parametrize("text_length", [0, 7])
def test_native_moss_layout_preserves_text_and_reference_tokens(model_factory, reference_mode, text_length):
    original = model_factory()
    observed = observe_blocks(original)
    patched = noise_patch(original, blocks="1")
    kwargs = {}
    refs = torch.arange(128 * 2 * 3, dtype=torch.float32).reshape(1, 128, 2, 3) + 100
    cache = torch.arange(1 * 3 * 2 * 6 * 16, dtype=torch.float32).reshape(1, 3, 2, 6, 16)
    refs_before, cache_before = refs.clone(), cache.clone()
    if reference_mode in ("index", "index_timestep_zero", "preprocess"):
        kwargs = {"ref_latents": [refs], "ref_latents_method": reference_mode}
    elif reference_mode == "cached":
        kwargs = {"ref_kv": cache}
    output = run_model(patched, text_length=text_length, **kwargs)
    block = original.model.diffusion_model.layers[1]
    direct = observed[1]
    native = block.calls[-1]["output"]
    expected_tokens = 15 + text_length + (6 if reference_mode in ("index", "index_timestep_zero") else 0)
    assert direct.shape == (2, expected_tokens, 16)
    assert not torch.equal(direct[:, :15], native[:, :15])
    torch.testing.assert_close(direct[:, 15:], native[:, 15:], rtol=0, atol=0)
    torch.testing.assert_close(refs, refs_before, rtol=0, atol=0)
    torch.testing.assert_close(cache, cache_before, rtol=0, atol=0)
    assert output.shape == (2, 128, 3, 5)


@pytest.mark.parametrize("height,width", [(1, 1), (3, 5), (8, 10)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_noise_uses_spatial_feature_shape_and_preserves_dtype(model_factory, height, width, dtype):
    original = model_factory()
    with patch.object(dit_noise, "generate_oneko_noise_for_tensor", wraps=dit_noise.generate_oneko_noise_for_tensor) as generate:
        output = run_model(noise_patch(original, blocks="0"), height=height, width=width, dtype=dtype)
    assert generate.call_count == 1
    feature = generate.call_args.args[0]
    assert feature.shape == (2, 16, height, width)
    assert feature.dtype == dtype
    assert feature.device.type == "cpu"
    assert output.dtype == dtype and output.device.type == "cpu"
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("noise_type", dit_noise.NOISE_TYPES)
def test_all_existing_noise_profiles_run_on_spatial_dit_features(model_factory, noise_type):
    original = model_factory()
    patched = noise_patch(original, blocks="1", noise_type=noise_type)
    output = run_model(patched, height=8, width=10)
    assert output.shape == (2, 128, 8, 10)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, run_model(patched, height=8, width=10), rtol=0, atol=0)


def test_seed_repeatability_sigma_and_original_model_isolation(model_factory):
    original = model_factory()
    original_options = copy.deepcopy(original.model_options)
    rng_state = torch.random.get_rng_state()
    first = noise_patch(original)
    second = noise_patch(original)
    assert first is not original and first.model is original.model
    torch.testing.assert_close(run_model(first), run_model(second), rtol=0, atol=0)
    assert not torch.equal(run_model(first), run_model(noise_patch(original, seed=124)))
    assert not torch.equal(run_model(first), run_model(first, sigma=0.6))
    assert original.model_options == original_options
    assert original.wrappers == {}
    assert torch.equal(torch.random.get_rng_state(), rng_state)


@pytest.mark.parametrize("start,end", [(0.2, 0.8), (0.8, 0.2)])
def test_noise_respects_sampling_window(model_factory, start, end):
    original = model_factory()
    patched = noise_patch(original, start_percent=start, end_percent=end)
    torch.testing.assert_close(run_model(patched, sigma=0.1), run_model(original, sigma=0.1), rtol=0, atol=0)
    torch.testing.assert_close(run_model(patched, sigma=0.9), run_model(original, sigma=0.9), rtol=0, atol=0)
    assert not torch.equal(run_model(patched), run_model(original))


def test_full_replacement_skips_original_and_its_noise(model_factory):
    original = model_factory()
    def skip(args, extra):
        return {"img": args["img"] + 0.75}
    original.set_model_patch_replace(skip, "dit", "single_block", 1)
    patched = noise_patch(original, blocks="1")
    with patch.object(dit_noise, "generate_oneko_noise_for_tensor", side_effect=AssertionError("skipped block must not generate noise")):
        torch.testing.assert_close(run_model(patched), run_model(original), rtol=0, atol=0)
    assert original.model.diffusion_model.layers[1].calls == []


def test_existing_wrappers_and_multiple_noise_nodes_compose(model_factory):
    original = model_factory()
    seen = []
    def existing(executor, *args, **kwargs):
        seen.append("existing")
        return executor(*args, **kwargs)
    original.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "existing", existing)
    first = noise_patch(original, blocks="0")
    second = noise_patch(first, blocks="1")
    output = run_model(second)
    assert seen == ["existing"]
    assert len(original.get_all_wrappers(WrappersMP.DIFFUSION_MODEL)) == 1
    assert len(first.get_all_wrappers(WrappersMP.DIFFUSION_MODEL)) == 2
    assert len(second.get_all_wrappers(WrappersMP.DIFFUSION_MODEL)) == 3
    assert not torch.equal(output, run_model(first))


def test_two_noise_nodes_on_same_block_execute_original_once(model_factory):
    original = model_factory()
    first = noise_patch(original, blocks="1", seed=10)
    second = noise_patch(first, blocks="1", seed=20)
    with patch.object(dit_noise, "generate_oneko_noise_for_tensor", wraps=dit_noise.generate_oneko_noise_for_tensor) as generate:
        output = run_model(second)
    assert generate.call_count == 2
    assert len(original.model.diffusion_model.layers[1].calls) == 1
    assert torch.isfinite(output).all()
    assert len(first.get_all_wrappers(WrappersMP.DIFFUSION_MODEL)) == 1
    assert len(second.get_all_wrappers(WrappersMP.DIFFUSION_MODEL)) == 2
    neutral = noise_patch(second, noise_strength=0)
    torch.testing.assert_close(output, run_model(neutral), rtol=0, atol=0)


def slg_result(patcher):
    options = runtime_options(patcher)
    patcher.model.current_patcher = patcher
    conditions = [[{"model_conds": {"c_crossattn": CONDRegular(torch.ones(1, length, 16) * value)}, "uuid": str(value)}]
                  for length, value in [(5, 2), (7, 4)]]
    return options["sampler_calc_cond_batch_function"]({
        "model": patcher.model, "input": torch.ones(1, 128, 3, 5), "conds": conditions,
        "sigma": torch.tensor([0.5]), "model_options": options,
    })


@pytest.mark.parametrize("order", ["noise-first", "slg-first"])
def test_native_slg_simple_skips_noise_with_original_in_both_node_orders(model_factory, order):
    original = model_factory()
    def slg(model):
        return SkipLayerGuidanceDiTSimple.execute(model, 0.0, 1.0, double_layers="", single_layers="1").result[0]
    baseline = slg(original)
    expected = slg_result(baseline)
    if order == "noise-first":
        combined = slg(noise_patch(original, blocks="1"))
    else:
        combined = noise_patch(slg(original), blocks="1")
    actual = slg_result(combined)
    assert not torch.equal(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    assert original.wrappers == {}
    assert "sampler_calc_cond_batch_function" not in original.model_options


def full_slg_result(patcher):
    options = runtime_options(patcher)
    patcher.model.current_patcher = patcher
    conditioning = [{"model_conds": {"c_crossattn": CONDRegular(torch.ones(1, 5, 16) * 2)}, "uuid": "positive"}]
    sample = torch.ones(1, 128, 3, 5)
    sigma = torch.tensor([0.5])
    (positive,) = comfy.samplers.calc_cond_batch(patcher.model, [conditioning], sample, sigma, options)
    result = positive.clone()
    for callback in options["sampler_post_cfg_function"]:
        result = callback({"model": patcher.model, "cond_denoised": positive, "cond": conditioning,
                           "denoised": result, "sigma": sigma, "input": sample, "model_options": options})
    return result, positive


@pytest.mark.parametrize("order", ["noise-first", "slg-first"])
def test_native_post_cfg_slg_preserves_skipped_branch_in_both_node_orders(model_factory, order):
    original = model_factory()
    scale = 2.0
    def slg(model):
        return SkipLayerGuidanceDiT.execute(model, scale, 0.0, 1.0,
                                           double_layers="", single_layers="1").result[0]
    baseline, native_positive = full_slg_result(slg(original))
    # Recover the skipped prediction from the real native post-CFG result.
    native_skipped = native_positive - (baseline - native_positive) / scale
    combined = slg(noise_patch(original, blocks="1")) if order == "noise-first" else noise_patch(slg(original), blocks="1")
    for block in original.model.diffusion_model.layers:
        block.calls.clear()
    actual, noisy_positive = full_slg_result(combined)
    assert not torch.equal(noisy_positive, native_positive)
    expected = noisy_positive + (noisy_positive - native_skipped) * scale
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert [len(block.calls) for block in original.model.diffusion_model.layers] == [2, 1, 2]
    assert original.wrappers == {}
    assert "sampler_post_cfg_function" not in original.model_options


@pytest.mark.parametrize("architecture", ["unet", "flux", "lookalike"])
def test_only_supported_moss_architecture_is_accepted(model_factory, architecture):
    if architecture == "unet":
        from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
        cls = UNetModel
    elif architecture == "flux":
        from comfy.ldm.flux.model import Flux
        cls = Flux
    else:
        cls = type("MOSSLookalike", (torch.nn.Module,), {})
    diffusion = cls.__new__(cls)
    torch.nn.Module.__init__(diffusion)
    diffusion.layers = torch.nn.ModuleList([RecordingBlock(i) for i in range(3)])
    original = ModelPatcher(ModelHost(diffusion), torch.device("cpu"), torch.device("cpu"))
    with pytest.raises(ValueError, match="supports MOSS only"):
        noise_patch(original)
    assert original.wrappers == {}


def test_invalid_noise_type_is_not_silently_replaced(model_factory):
    with pytest.raises(ValueError, match="Unknown noise type"):
        noise_patch(model_factory(), noise_type="unknown-profile")


def test_node_import_and_registration_work_without_native_moss_support():
    import builtins
    import importlib.util
    import sys

    real_import = builtins.__import__
    def without_moss(name, *args, **kwargs):
        if name == "comfy.ldm.moss_image" or name.startswith("comfy.ldm.moss_image."):
            raise ImportError("MOSS is unavailable in this Core")
        return real_import(name, *args, **kwargs)

    name = "Oneko_ComfyUI_Nodes.nodes._dit_legacy_import_test"
    spec = importlib.util.spec_from_file_location(name, dit_noise.__file__)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {name: module}), patch.object(builtins, "__import__", side_effect=without_moss):
        spec.loader.exec_module(module)
        node = module.NODE_CLASS_MAPPINGS["OnekoDiTBlockNoisePatch"]
        assert node is module.OnekoDiTBlockNoisePatch
        assert module.NODE_DISPLAY_NAME_MAPPINGS["OnekoDiTBlockNoisePatch"]
        with pytest.raises(ValueError, match="requires ComfyUI with native MOSS support.*Update ComfyUI"):
            node.execute(object())
