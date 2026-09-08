import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


from tests.paths import COMFY_ROOT, ONEKO_ROOT
CUSTOM_NODES_ROOT = COMFY_ROOT / "custom_nodes"

WORKFLOW_PATH = ONEKO_ROOT / "examples" / "oneko_example_07_simple_universal_ksampler.json"

for path in (str(COMFY_ROOT), str(CUSTOM_NODES_ROOT), str(ONEKO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from Oneko_ComfyUI_Nodes.nodes import noise_sampler_core  # noqa: E402
from Oneko_ComfyUI_Nodes.nodes.universal_noise_sampler import (  # noqa: E402
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)


class _FakeModel:
    def process_latent_out(self, latent):
        return latent


class _FakeModelPatcher:
    load_device = "cpu"
    model = _FakeModel()


class _FakeGuider:
    model_patcher = _FakeModelPatcher()

    def __init__(self, should_raise=False):
        self.should_raise = should_raise

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=True, seed=0):
        if self.should_raise:
            raise RuntimeError("sampling failed")
        return latent_image + noise


class _FakeNoise:
    seed = 123

    def generate_noise(self, latent):
        return torch.zeros_like(latent["samples"])


class SamplerPreviewTests(unittest.TestCase):
    def setUp(self):
        self.latent = {"samples": torch.zeros(1, 4, 8, 8)}
        self.sigmas = torch.tensor([1.0, 0.5, 0.0])

    def _sample(self, preview_method="default", should_raise=False):
        return noise_sampler_core.sample_custom_advanced(
            _FakeGuider(should_raise=should_raise),
            sampler=object(),
            sigmas=self.sigmas,
            latent_image=self.latent,
            noise=_FakeNoise(),
            noise_seed=123,
            preview_method=preview_method,
        )

    @mock.patch("Oneko_ComfyUI_Nodes.nodes.noise_sampler_core.comfy.model_management.intermediate_device", return_value="cpu")
    @mock.patch("Oneko_ComfyUI_Nodes.nodes.noise_sampler_core.comfy.sample.fix_empty_latent_channels", side_effect=lambda model, samples, *args: samples)
    @mock.patch("Oneko_ComfyUI_Nodes.nodes.noise_sampler_core.latent_preview.prepare_callback", return_value=lambda *args: None)
    @mock.patch("Oneko_ComfyUI_Nodes.nodes.noise_sampler_core.latent_preview.set_preview_method")
    def test_default_preview_method_does_not_override_global_setting(
        self,
        set_preview_method,
        prepare_callback,
        fix_empty_latent_channels,
        intermediate_device,
    ):
        out, denoised, seed = self._sample("default")

        self.assertEqual(seed, 123)
        self.assertEqual(out["samples"].device.type, "cpu")
        self.assertEqual(denoised["samples"].device.type, "cpu")
        set_preview_method.assert_not_called()
        prepare_callback.assert_called_once()

    @mock.patch("Oneko_ComfyUI_Nodes.nodes.noise_sampler_core.comfy.model_management.intermediate_device", return_value="cpu")
    @mock.patch("Oneko_ComfyUI_Nodes.nodes.noise_sampler_core.comfy.sample.fix_empty_latent_channels", side_effect=lambda model, samples, *args: samples)
    @mock.patch("Oneko_ComfyUI_Nodes.nodes.noise_sampler_core.latent_preview.prepare_callback", return_value=lambda *args: None)
    @mock.patch("Oneko_ComfyUI_Nodes.nodes.noise_sampler_core.latent_preview.set_preview_method")
    def test_preview_overrides_restore_after_sampling(
        self,
        set_preview_method,
        prepare_callback,
        fix_empty_latent_channels,
        intermediate_device,
    ):
        for method in ("latent2rgb", "taesd", "none"):
            with self.subTest(method=method):
                set_preview_method.reset_mock()
                self._sample(method)
                self.assertEqual(
                    set_preview_method.mock_calls,
                    [mock.call(method), mock.call(None)],
                )

    @mock.patch("Oneko_ComfyUI_Nodes.nodes.noise_sampler_core.comfy.sample.fix_empty_latent_channels", side_effect=lambda model, samples, *args: samples)
    @mock.patch("Oneko_ComfyUI_Nodes.nodes.noise_sampler_core.latent_preview.prepare_callback", return_value=lambda *args: None)
    @mock.patch("Oneko_ComfyUI_Nodes.nodes.noise_sampler_core.latent_preview.set_preview_method")
    def test_preview_override_restores_when_sampling_raises(
        self,
        set_preview_method,
        prepare_callback,
        fix_empty_latent_channels,
    ):
        with self.assertRaises(RuntimeError):
            self._sample("latent2rgb", should_raise=True)

        self.assertEqual(
            set_preview_method.mock_calls,
            [mock.call("latent2rgb"), mock.call(None)],
        )


class UniversalKSamplerRegistrationTests(unittest.TestCase):
    def test_universal_ksampler_is_registered(self):
        self.assertIn("OnekoUniversalKSampler", NODE_CLASS_MAPPINGS)
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["OnekoUniversalKSampler"],
            "Universal KSampler (Oneko)",
        )

    def test_universal_ksampler_has_preview_method_input(self):
        inputs = NODE_CLASS_MAPPINGS["OnekoUniversalKSampler"].INPUT_TYPES()["required"]
        self.assertIn("preview_method", inputs)
        self.assertEqual(inputs["preview_method"][0], noise_sampler_core.PREVIEW_METHODS)


class SimpleUniversalKSamplerWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
        cls.nodes = {int(node["id"]): node for node in cls.workflow["nodes"]}
        cls.links = {int(link[0]): link for link in cls.workflow["links"]}

    def nodes_of_type(self, node_type):
        return [node for node in self.nodes.values() if node["type"] == node_type]

    def test_simple_workflow_does_not_require_advanced_sampler_plumbing(self):
        self.assertEqual(self.nodes_of_type("CFGGuider"), [])
        self.assertEqual(self.nodes_of_type("KSamplerSelect"), [])
        self.assertEqual(self.nodes_of_type("BasicScheduler"), [])
        self.assertEqual(len(self.nodes_of_type("OnekoUniversalKSampler")), 1)

    def test_simple_sampler_receives_core_inputs(self):
        sampler = self.nodes_of_type("OnekoUniversalKSampler")[0]
        input_types = {input_data["name"]: input_data["type"] for input_data in sampler["inputs"]}

        self.assertEqual(input_types["model"], "MODEL")
        self.assertEqual(input_types["positive"], "CONDITIONING")
        self.assertEqual(input_types["negative"], "CONDITIONING")
        self.assertEqual(input_types["latent_image"], "LATENT")

    def test_simple_sampler_output_reaches_vae_decode(self):
        sampler = self.nodes_of_type("OnekoUniversalKSampler")[0]
        decode = self.nodes_of_type("VAEDecode")[0]
        output_link = sampler["outputs"][0]["links"][0]
        link = self.links[output_link]

        self.assertEqual((int(link[1]), int(link[2])), (sampler["id"], 0))
        self.assertEqual((int(link[3]), int(link[4])), (decode["id"], 0))
        self.assertEqual(link[5], "LATENT")


if __name__ == "__main__":
    unittest.main()
