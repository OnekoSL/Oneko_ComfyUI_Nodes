import json
import sys
import unittest
from pathlib import Path
from unittest import mock


from tests.paths import COMFY_ROOT, ONEKO_ROOT
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

from Oneko_ComfyUI_Nodes.nodes import ollama_prompt_refiner as refiner


class FakeResponse:
    def __init__(self, response="", **extra):
        self.data = json.dumps({"done": True, "response": response, **extra}).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.data


class ReasoningModelRequestTests(unittest.TestCase):
    def test_reka_uses_schema_first_generation_for_the_four_verified_profiles(self):
        model = "autoren-reka-flash-3-21b-reasoning-q4:latest"
        for profile in ("krea2", "z_image", "wan2_2_video", "pony_v7"):
            with self.subTest(profile=profile):
                self.assertEqual(
                    refiner._generation_request_settings(model, profile),
                    {"reasoning": False, "num_predict": 900},
                )

        self.assertEqual(
            refiner._generation_request_settings(model, "anima"),
            {"reasoning": True, "num_predict": 1400},
        )
        self.assertEqual(
            refiner._generation_request_settings("qwen3:8b", "z_image"),
            {"reasoning": True, "num_predict": 1400},
        )

    def request_payload(
        self,
        model,
        response='{"answer":"ok"}',
        num_predict=500,
        reasoning=True,
        system_instructions=refiner.SYSTEM_INSTRUCTIONS,
    ):
        with mock.patch.object(
            refiner.urllib.request,
            "urlopen",
            return_value=FakeResponse(response),
        ) as urlopen:
            result = refiner._request_ollama(
                "http://127.0.0.1:11434",
                model,
                "Return JSON.",
                7,
                0.6,
                0.95,
                30,
                8192,
                output_schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
                system_instructions=system_instructions,
                num_predict=num_predict,
                reasoning=reasoning,
            )
        request = urlopen.call_args.args[0]
        return result, json.loads(request.data.decode("utf-8"))

    def test_reka_allows_inline_reasoning_before_json(self):
        response = '<reasoning>Check the requirement.</reasoning>\n{"answer":"ok"}'
        result, payload = self.request_payload(
            "autoren-reka-flash-3-21b-reasoning:latest",
            response=response,
        )

        self.assertEqual(result, '{"answer":"ok"}')
        self.assertNotIn("format", payload)
        self.assertNotIn("system", payload)
        self.assertIn("First think briefly inside <reasoning>", payload["prompt"])
        self.assertIn("below 400 tokens", payload["prompt"])
        self.assertEqual(payload["options"]["stop"], ["<sep>", "<|endoftext|>"])
        self.assertEqual(payload["options"]["top_k"], 1024)
        self.assertEqual(payload["options"]["num_predict"], 1000)
        self.assertIn('"answer":{"type":"string"}', payload["prompt"])

    def test_reka_structured_stage_can_disable_slow_reasoning(self):
        result, payload = self.request_payload(
            "autoren-reka-flash-3-21b-reasoning-q4:latest",
            reasoning=False,
            system_instructions="Translate German.",
        )

        self.assertEqual(result, '{"answer":"ok"}')
        self.assertIn("format", payload)
        self.assertIn("without reasoning", payload["prompt"])
        self.assertNotIn("First think briefly", payload["prompt"])
        self.assertEqual(payload["options"]["num_predict"], 500)
        self.assertIn("Translate German.", payload["prompt"])
        self.assertLess(
            payload["prompt"].index("System instructions:"), payload["prompt"].index("Task instructions:")
        )

    def test_regular_models_keep_structured_output_contract(self):
        _result, payload = self.request_payload("qwen3:8b")

        self.assertIn("format", payload)
        self.assertIn("system", payload)
        self.assertNotIn("top_k", payload["options"])
        self.assertEqual(payload["options"]["num_predict"], 500)

    def test_native_thinking_models_do_not_constrain_reasoning_with_json_grammar(self):
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        responses = (
            FakeResponse(capabilities=["completion", "thinking"]),
            FakeResponse('<think>Check the requirement.</think>\n{"answer":"ok"}'),
        )
        with mock.patch.object(
            refiner.urllib.request,
            "urlopen",
            side_effect=responses,
        ) as urlopen:
            result = refiner._request_ollama(
                "http://127.0.0.1:11434",
                "qwen3:8b",
                "Return JSON.",
                7,
                0.6,
                0.95,
                30,
                8192,
                output_schema=schema,
                num_predict=500,
            )

        self.assertEqual(result, '{"answer":"ok"}')
        show_payload = json.loads(urlopen.call_args_list[0].args[0].data.decode("utf-8"))
        generate_payload = json.loads(urlopen.call_args_list[1].args[0].data.decode("utf-8"))
        self.assertEqual(show_payload, {"model": "qwen3:8b"})
        self.assertTrue(generate_payload["think"])
        self.assertNotIn("format", generate_payload)
        self.assertEqual(generate_payload["options"]["num_predict"], 1000)

    def test_structured_stage_explicitly_disables_native_thinking(self):
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        responses = (
            FakeResponse(capabilities=["completion", "thinking"]),
            FakeResponse('{"answer":"ok"}'),
        )
        with mock.patch.object(
            refiner.urllib.request,
            "urlopen",
            side_effect=responses,
        ) as urlopen:
            result = refiner._request_ollama(
                "http://127.0.0.1:11434",
                "qwen3:8b",
                "Return JSON.",
                7,
                0.6,
                0.95,
                30,
                8192,
                output_schema=schema,
                num_predict=500,
                reasoning=False,
            )

        self.assertEqual(result, '{"answer":"ok"}')
        generate_payload = json.loads(urlopen.call_args_list[1].args[0].data.decode("utf-8"))
        self.assertFalse(generate_payload["think"])
        self.assertEqual(generate_payload["format"], schema)
        self.assertEqual(generate_payload["options"]["num_predict"], 500)


if __name__ == "__main__":
    unittest.main()
