"""Per-call USDU noise/reference hooks, including native Core sampling parity."""

import copy
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest
import torch

import nodes as core_nodes
import comfy.patcher_extension as extension
import comfy.sample
import comfy.samplers
from comfy.patcher_extension import WrappersMP, WrapperExecutor
from Oneko_ComfyUI_Nodes.nodes import hiresfix_tiled as hires
from Oneko_ComfyUI_Nodes.nodes import noise_sampler_core as noise_core
from tests.test_moss_hiresfix_integration import make_model
from tests.test_moss_sampling_integration import moss_model, inputs


@pytest.mark.parametrize("profile", noise_core.UNIVERSAL_NOISE_PROFILES)
def test_tile_wrapper_preserves_profile_seed_strength_and_arguments(profile):
    latent = torch.linspace(.1, .9, 2 * 4 * 5 * 7).reshape(2, 4, 5, 7)
    mask = torch.ones(2, 1, 5, 7)
    sigmas = torch.tensor([1., .5, 0.])
    sampler, callback = object(), object()
    seed = 0xffffffffffffffff
    expected = noise_core.make_noise_generator(seed, "cpu", profile, .55, .25).generate_noise({"samples": latent})
    wrapper = hires._make_usdu_sampling_wrapper("cpu", profile, .55, .25, warn_missing_noise_mask=True)
    conds = {"positive": [{"marker": "unchanged"}], "negative": []}
    guider = SimpleNamespace(conds=conds)

    def sample(noise, actual_latent, actual_sampler, actual_sigmas, denoise_mask,
               actual_callback, disable_pbar, actual_seed, **kwargs):
        torch.testing.assert_close(noise, expected, rtol=0, atol=0)
        assert actual_latent is latent and actual_sampler is sampler and actual_sigmas is sigmas
        assert denoise_mask is mask and actual_callback is callback
        assert disable_pbar is True and actual_seed == seed
        assert kwargs == {"latent_shapes": [latent.shape]}
        assert guider.conds is conds
        return actual_latent + noise

    before = torch.random.get_rng_state().clone()
    result = WrapperExecutor.new_class_executor(sample, guider, [wrapper]).execute(
        torch.zeros_like(latent), latent, sampler, sigmas, mask, callback, True, seed,
        latent_shapes=[latent.shape],
    )
    torch.testing.assert_close(result, latent + expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    assert not wrapper.missing_noise_mask


@pytest.mark.parametrize("fail_first", [False, True])
def test_overlapping_tile_calls_keep_noise_references_and_failures_isolated(fail_first):
    barrier = Barrier(2)
    original_sample = comfy.sample.sample
    model = make_model()
    original_options = copy.deepcopy(model.model_options)
    original_wrappers = copy.deepcopy(model.wrappers)
    latent = torch.ones(2, 128, 3, 4)
    states = []

    def run(index):
        seed = 37 + index
        profile = ("pink", "pony_v7_balanced")[index]
        reference = torch.full((1, 128, 2, 3), .2 + index)
        wrapper = hires._make_usdu_sampling_wrapper(
            "cpu", profile, .7, .35, {"samples": reference}, warn_missing_noise_mask=True,
        )
        patched = hires._with_tile_sampling(model, wrapper)
        conds = {"positive": [{"reference_latents": [torch.zeros(1)]}], "negative": []}
        guider = SimpleNamespace(conds=conds)
        expected = noise_core.make_noise_generator(seed, "cpu", profile, .7, .35).generate_noise({"samples": latent})

        def sample(noise, actual_latent, *args, **kwargs):
            barrier.wait(timeout=10)
            assert comfy.sample.sample is original_sample
            torch.testing.assert_close(noise, expected, rtol=0, atol=0)
            refs = guider.conds["positive"][0]["reference_latents"]
            assert len(refs) == 2
            torch.testing.assert_close(refs[-1], reference.repeat_interleave(2, dim=0), rtol=0, atol=0)
            assert len(conds["positive"][0]["reference_latents"]) == 1
            if fail_first and index == 0:
                raise RuntimeError("isolated tile failure")
            return actual_latent + noise

        wrappers = extension.get_all_wrappers(WrappersMP.OUTER_SAMPLE, patched.model_options, is_model_options=True)
        try:
            return WrapperExecutor.new_class_executor(sample, guider, wrappers).execute(
                torch.zeros_like(latent), latent, object(), torch.tensor([1., 0.]),
                None if index == 0 else torch.ones_like(latent), None, True, seed,
            )
        finally:
            assert guider.conds is conds
            assert wrapper.missing_noise_mask == (index == 0)
            states.append(index)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, index) for index in range(2)]
        if fail_first:
            with pytest.raises(RuntimeError, match="isolated tile failure"):
                futures[0].result(timeout=15)
        else:
            assert torch.isfinite(futures[0].result(timeout=15)).all()
        assert torch.isfinite(futures[1].result(timeout=15)).all()
    assert sorted(states) == [0, 1]
    assert comfy.sample.sample is original_sample
    assert model.model_options == original_options and model.wrappers == original_wrappers


@pytest.mark.parametrize("sampler_name", ["euler", "dpm_2", "euler_ancestral"])
@pytest.mark.parametrize("denoise", [0., .4, 1.])
@pytest.mark.parametrize("profile", ["gaussian", "pink", "pony_v7_balanced"])
def test_native_sampling_and_decoded_pixels_match_previous_noise_handoff(moss_model, sampler_name, denoise, profile):
    latent, positive, negative, _ = inputs()
    latent.pop("batch_index")  # USDU's VAEEncode returns ordinary image tiles.
    reference = {"samples": torch.full((1, 128, 2, 3), .3)}
    expected_positive = hires._first_output(hires.ReferenceLatent.execute(
        positive, {"samples": reference["samples"].repeat_interleave(2, dim=0)},
    ))
    seed = 514
    expected_noise = noise_core.make_noise_generator(seed, "cpu", profile, .7, .35).generate_noise(latent)
    expected = comfy.sample.sample(
        moss_model, expected_noise, 4, 2.5, sampler_name, "simple", expected_positive,
        negative, latent["samples"], denoise=denoise, noise_mask=latent["noise_mask"], seed=seed,
    )
    wrapper = hires._make_usdu_sampling_wrapper("cpu", profile, .7, .35, reference)
    patched = hires._with_tile_sampling(moss_model, wrapper)
    actual = core_nodes.common_ksampler(
        patched, seed, 4, 2.5, sampler_name, "simple", positive, negative, latent, denoise=denoise,
    )[0]["samples"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # A deterministic CPU decoder converts the full latent equality to pixels.
    from tests.test_moss_hiresfix_integration import SmallVAE
    torch.testing.assert_close(SmallVAE(16).decode(actual), SmallVAE(16).decode(expected), rtol=0, atol=0)
    assert len(positive[0][1]["reference_latents"]) == 1
    assert not moss_model.model_options["transformer_options"].get("wrappers")


def test_existing_outer_wrappers_see_oneko_noise_and_keep_their_changes(moss_model, monkeypatch):
    latent, positive, negative, _ = inputs()
    # Remove batch indices before computing the tile-equivalent expectation.
    latent.pop("batch_index")
    expected = noise_core.make_noise_generator(51, "cpu", "pink", .7, .35).generate_noise(latent)
    order = []

    def options_observer(executor, noise, *args, **kwargs):
        order.append("options")
        torch.testing.assert_close(noise, expected, rtol=0, atol=0)
        return executor(noise + 1, *args, **kwargs)

    def patcher_observer(executor, noise, *args, **kwargs):
        order.append("patcher")
        torch.testing.assert_close(noise, expected + 1, rtol=0, atol=0)
        return executor(noise * 2, *args, **kwargs)

    extension.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, "options_observer", options_observer,
                                   moss_model.model_options, is_model_options=True)
    moss_model.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, "patcher_observer", patcher_observer)
    before = copy.deepcopy(moss_model.model_options)
    wrapper = hires._make_usdu_sampling_wrapper("cpu", "pink", .7, .35)
    patched = hires._with_tile_sampling(moss_model, wrapper)

    def outer_sample(self, noise, latent_image, *args, **kwargs):
        order.append("sample")
        torch.testing.assert_close(noise, (expected + 1) * 2, rtol=0, atol=0)
        return latent_image

    monkeypatch.setattr(comfy.samplers.CFGGuider, "outer_sample", outer_sample)
    core_nodes.common_ksampler(patched, 51, 4, 2.5, "euler", "simple", positive, negative, latent)
    assert order == ["options", "patcher", "sample"]
    assert moss_model.model_options == before
