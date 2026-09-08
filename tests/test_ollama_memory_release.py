import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


from tests.paths import COMFY_ROOT, ONEKO_ROOT
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

from Oneko_ComfyUI_Nodes.nodes import ollama_prompt_refiner as refiner
from Oneko_ComfyUI_Nodes.nodes import ollama_vision_captioner as captioner


class FakeResponse:
    def __init__(self, data):
        self.data = json.dumps(data).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.data


class OllamaMemoryReleaseTests(unittest.TestCase):
    def test_nodes_default_to_4096_context_and_unload_enabled(self):
        prompt_inputs = refiner.OnekoOllamaPromptRefiner.INPUT_TYPES()
        vision_inputs = captioner.OnekoOllamaVisionCaptioner.INPUT_TYPES()
        self.assertEqual(prompt_inputs["required"]["context_length"][1]["default"], "4096")
        self.assertEqual(vision_inputs["required"]["context_length"][1]["default"], "4096")
        self.assertTrue(prompt_inputs["optional"]["unload_after_run"][1]["default"])
        self.assertTrue(vision_inputs["optional"]["unload_after_run"][1]["default"])

    def test_unload_request_uses_keep_alive_zero(self):
        response = FakeResponse({"done": True, "done_reason": "unload"})
        with mock.patch.object(refiner.urllib.request, "urlopen", return_value=response) as urlopen:
            refiner._unload_ollama_model(
                "http://127.0.0.1:11434",
                "test-model",
                30,
            )
        request = urlopen.call_args.args[0]
        self.assertEqual(
            json.loads(request.data.decode("utf-8")),
            {"model": "test-model", "keep_alive": 0, "stream": False},
        )
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 10)

    def test_prompt_refiner_unloads_once_after_complete_run(self):
        events = []
        result = ("positive", "negative", "report", "base", "foreground", "background", "{}", "{}")
        node = refiner.OnekoOllamaPromptRefiner()
        with mock.patch.object(
            node,
            "_refine",
            side_effect=lambda *args: events.append("run") or result,
        ), mock.patch.object(
            refiner,
            "_unload_after_run",
            side_effect=lambda *args: events.append("unload"),
        ) as unload:
            self.assertEqual(
                node.refine(
                    "idea",
                    "http://127.0.0.1:11434",
                    "test-model",
                    "krea2",
                    0,
                    0.4,
                    0.9,
                    430,
                    30,
                ),
                result,
            )
        self.assertEqual(events, ["run", "unload"])
        unload.assert_called_once_with(
            "http://127.0.0.1:11434",
            "test-model",
            30,
            True,
        )

    def test_prompt_refiner_still_unloads_after_failure(self):
        node = refiner.OnekoOllamaPromptRefiner()
        with mock.patch.object(node, "_refine", side_effect=RuntimeError("failed")), mock.patch.object(
            refiner, "_unload_after_run"
        ) as unload:
            with self.assertRaisesRegex(RuntimeError, "failed"):
                node.refine(
                    "idea",
                    "http://127.0.0.1:11434",
                    "test-model",
                    "krea2",
                    0,
                    0.4,
                    0.9,
                    430,
                    30,
                )
        unload.assert_called_once_with(
            "http://127.0.0.1:11434",
            "test-model",
            30,
            True,
        )

    def test_vision_captioner_unloads_after_complete_run(self):
        result = ("caption", "tags", "seed", "report", "hires")
        node = captioner.OnekoOllamaVisionCaptioner()
        with mock.patch.object(node, "_caption", return_value=result), mock.patch.object(
            captioner, "_unload_after_run"
        ) as unload:
            self.assertEqual(
                node.caption(
                    np.zeros((1, 8, 8, 3), dtype=np.float32),
                    "http://127.0.0.1:11434",
                    "vision-model",
                    "refiner_seed",
                    0,
                    0.25,
                    0.9,
                    30,
                ),
                result,
            )
        unload.assert_called_once_with(
            "http://127.0.0.1:11434",
            "vision-model",
            30,
            True,
        )

    def test_disabled_release_does_not_send_unload_request(self):
        with mock.patch.object(refiner, "_unload_ollama_model") as unload:
            refiner._unload_after_run(
                "http://127.0.0.1:11434",
                "test-model",
                30,
                False,
            )
        unload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
