import json
import sys
import unittest
from pathlib import Path
from unittest import mock


from tests.paths import COMFY_ROOT, ONEKO_ROOT
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

from Oneko_ComfyUI_Nodes.nodes import ollama_prompt_refiner as refiner


def refine_kwargs(prompt_mode="strict"):
    return {
        "word_salad": "red dragon mountain valley",
        "ollama_url": "http://127.0.0.1:11434",
        "ollama_model": "test-model",
        "target_profile": "pony_v7",
        "seed": 10,
        "temperature": 0.45,
        "top_p": 0.9,
        "style_cluster": 430,
        "timeout_seconds": 30,
        "context_length": 8192,
        "style_anchor": "",
        "fallback_mode": "adaptive",
        "pipeline_mode": "single",
        "unload_after_run": False,
        "prompt_mode": prompt_mode,
    }


class PromptModeTests(unittest.TestCase):
    def test_interface_defaults_to_strict_and_offers_creative(self):
        choices, options = refiner.OnekoOllamaPromptRefiner.INPUT_TYPES()["optional"]["prompt_mode"]
        self.assertEqual(choices, ("strict", "creative"))
        self.assertEqual(options["default"], "strict")

    def test_creative_mode_adds_freedom_without_replacing_fixed_requirements(self):
        prompt = refiner._build_generation_prompt(
            "pony_v7",
            "red dragon mountain valley",
            "watercolor",
            430,
            prompt_mode="creative",
        )
        self.assertIn("Freely invent compatible poses", prompt)
        self.assertIn("never replace the main subject", prompt)
        self.assertIn("style anchor", prompt.lower())

        planner = refiner._build_planner_prompt(
            "pony_v7",
            "red dragon mountain valley",
            "watercolor",
            prompt_mode="creative",
        )
        self.assertIn("Freely invent compatible actions", planner)
        self.assertNotIn("Do not invent a new subject, setting, action", planner)

    def test_creative_mode_broadens_sampling_but_strict_keeps_values(self):
        self.assertEqual(refiner._prompt_mode_sampling("strict", 0.45, 0.9), (0.45, 0.9))
        self.assertEqual(refiner._prompt_mode_sampling("creative", 0.45, 0.9), (0.8, 0.95))
        self.assertEqual(refiner._prompt_mode_sampling("creative", 1.1, 0.98), (1.1, 0.98))

    def test_single_creative_request_uses_mode_prompt_sampling_and_system(self):
        response = json.dumps(
            {
                "base_prompt": "painterly fantasy",
                "foreground_prompt": "a red dragon flies",
                "background_prompt": "mountains fill the valley",
                "negative": "bad anatomy",
                "report": "Built a creative dragon prompt.",
            }
        )
        result = (
            "a red dragon flies above mountains",
            "bad anatomy",
            "Built a creative dragon prompt.",
            "painterly fantasy",
            "a red dragon flies",
            "mountains fill the valley",
        )
        node = refiner.OnekoOllamaPromptRefiner()
        language_inputs = refiner._language_source_values(
            "red dragon mountain valley", "", "", "", ""
        )
        with mock.patch.object(
            refiner,
            "_prepare_language_inputs",
            return_value=(language_inputs, "english", False),
        ), mock.patch.object(
            refiner, "_request_ollama", return_value=response
        ) as request, mock.patch.object(refiner, "_postprocess_result", return_value=result):
            node.refine(**refine_kwargs("creative"))

        call = request.call_args
        self.assertEqual(call.args[4:6], (0.8, 0.95))
        self.assertIn("Creative mode", call.args[2])
        self.assertIn("Creative mode", call.kwargs["system_instructions"])

    def test_prompt_mode_participates_in_cache_hash(self):
        strict = refiner.OnekoOllamaPromptRefiner.IS_CHANGED(**refine_kwargs("strict"))
        creative = refiner.OnekoOllamaPromptRefiner.IS_CHANGED(**refine_kwargs("creative"))
        self.assertNotEqual(strict, creative)


if __name__ == "__main__":
    unittest.main()
