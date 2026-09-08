import json
import sys
import unittest
from pathlib import Path
from unittest import mock


from tests.paths import COMFY_ROOT, ONEKO_ROOT
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

from Oneko_ComfyUI_Nodes.nodes import ollama_prompt_refiner as refiner  # noqa: E402


def passthrough_language_inputs(word_salad, left, right, top, bottom, *_args):
    return refiner._language_source_values(word_salad, left, right, top, bottom), "english", False


def plan_data(required=None, avoid=None):
    return {
        "subject": "a red dragon",
        "action": "flying",
        "environment": "mountain valley",
        "style": "illustration",
        "composition": "wide view",
        "lighting": "moonlight",
        "camera": "low angle",
        "subject_details": ["red scales"],
        "spatial_relations": [],
        "required_elements": list(required or []),
        "avoid": list(avoid or []),
        "discarded_terms": ["spreadsheet"],
    }


def compiler_data():
    return {
        "base_prompt": "illustration with moonlight",
        "foreground_prompt": "a red dragon flies above the valley",
        "background_prompt": "mountains surround the scene",
        "negative": "bad anatomy",
        "report": "Built the requested prompt",
    }


def review_data(needs_revision=False):
    return {
        "all_required_preserved": not needs_revision,
        "missing_elements": ["red dragon"] if needs_revision else [],
        "contradictions": [],
        "unwanted_additions": [],
        "profile_violations": [],
        "needs_revision": needs_revision,
        "summary": "Revision required" if needs_revision else "Prompt is complete",
    }


def refine_kwargs(profile="pony_v7", pipeline_mode="plan_compile", fallback_mode="strict"):
    return {
        "word_salad": "red dragon mountain valley moonlight spreadsheet",
        "ollama_url": "http://127.0.0.1:11434",
        "ollama_model": "test-model",
        "target_profile": profile,
        "seed": 10,
        "temperature": 0.45,
        "top_p": 0.9,
        "style_cluster": 430,
        "timeout_seconds": 30,
        "context_length": 8192,
        "style_anchor": "",
        "left": "",
        "right": "",
        "top": "",
        "bottom": "",
        "fallback_mode": fallback_mode,
        "pipeline_mode": pipeline_mode,
        "unload_after_run": False,
    }


class PipelineSchemaTests(unittest.TestCase):
    def test_interface_adds_pipeline_mode_and_appends_json_outputs(self):
        choices, options = refiner.OnekoOllamaPromptRefiner.INPUT_TYPES()["optional"]["pipeline_mode"]
        self.assertEqual(choices, ("single", "plan_compile", "plan_compile_review"))
        self.assertEqual(options["default"], "single")
        self.assertEqual(refiner.OnekoOllamaPromptRefiner.RETURN_NAMES[:6], refiner.OUTPUT_KEYS)
        self.assertEqual(refiner.OnekoOllamaPromptRefiner.RETURN_NAMES[6:], ("plan_json", "review_json"))

    def test_plan_validation_normalizes_lists_and_rejects_wrong_types(self):
        data = plan_data(required=["dragon", "dragon", "  moonlight  "])
        plan = refiner._validate_prompt_plan(data)
        self.assertEqual(plan["required_elements"], ["dragon", "moonlight"])
        data["avoid"] = {"unexpected": "object"}
        with self.assertRaisesRegex(ValueError, "avoid must be an array"):
            refiner._validate_prompt_plan(data)

    def test_plan_validation_recovers_stringified_list_fields(self):
        data = plan_data()
        data["subject_details"] = "red scales, sharp horns"
        data["avoid"] = ""
        plan = refiner._validate_prompt_plan(data)
        self.assertEqual(plan["subject_details"], ["red scales", "sharp horns"])
        self.assertEqual(plan["avoid"], [])

    def test_fixed_style_and_spatial_inputs_are_forced_into_plan(self):
        plan = refiner._enforce_fixed_plan_inputs(
            refiner._validate_prompt_plan(plan_data()),
            "watercolor illustration",
            left="brass telescope",
        )
        required = " ".join(plan["required_elements"]).lower()
        self.assertIn("watercolor", required)
        self.assertIn("brass telescope", required)
        self.assertIn("left: brass telescope", plan["spatial_relations"])

    def test_planner_subject_and_required_elements_must_be_source_grounded(self):
        bad_subject = refiner._validate_prompt_plan(plan_data())
        bad_subject["subject"] = "pony_v7"
        with self.assertRaisesRegex(ValueError, "subject is not grounded"):
            refiner._validate_plan_source(bad_subject, "red dragon mountain valley", "")
        bad_required = refiner._validate_prompt_plan(plan_data(required=["golden necklace"]))
        with self.assertRaisesRegex(ValueError, "required elements are not grounded"):
            refiner._validate_plan_source(bad_required, "red dragon mountain valley", "")

    def test_planner_requires_source_verbatim_subject_and_requirements(self):
        prompt = refiner._build_planner_prompt(
            "anima",
            "furry anthro orangutan waterfall",
            "",
        )
        self.assertIn("copied verbatim from the source", prompt)
        self.assertIn("Do not paraphrase", prompt)
        self.assertIn("required_elements source-verbatim", prompt)

    def test_local_validation_finds_missing_required_and_avoided_content(self):
        plan = refiner._validate_prompt_plan(plan_data(required=["dragon", "crystal crown"], avoid=["city"]))
        result = (
            "a red dragon flies above a city",
            "bad anatomy",
            "ok",
            "illustration",
            "a red dragon flies",
            "a city below",
        )
        issues = refiner._local_pipeline_issues("pony_v7", plan, result, style_anchor="watercolor")
        self.assertIn("missing required element: crystal crown", issues)
        self.assertIn("missing required element: watercolor", issues)
        self.assertIn("avoided element appears in positive: city", issues)

    def test_local_validation_requires_planned_subject_in_foreground(self):
        plan = refiner._validate_prompt_plan(plan_data())
        result = (
            "a copper-haired archivist studies a crystal while a red dragon appears in the distance",
            "bad anatomy",
            "ok",
            "illustration",
            "a copper-haired archivist studies a crystal",
            "a red dragon flies above a mountain valley",
        )
        issues = refiner._local_pipeline_issues("anima", plan, result)
        self.assertIn(
            "planned subject missing from foreground_prompt: a red dragon",
            issues,
        )

    def test_review_findings_force_consistent_revision_flags(self):
        review = review_data()
        review["profile_violations"] = ["foreign main subject"]
        normalized = refiner._normalize_review_consistency(review)
        self.assertTrue(normalized["all_required_preserved"])
        self.assertTrue(normalized["needs_revision"])

        review = review_data()
        review["missing_elements"] = ["red dragon"]
        normalized = refiner._normalize_review_consistency(review)
        self.assertFalse(normalized["all_required_preserved"])
        self.assertTrue(normalized["needs_revision"])


class PipelineExecutionTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(
            refiner,
            "_prepare_language_inputs",
            side_effect=passthrough_language_inputs,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    clean_result = (
        "a red dragon flies above a mountain valley",
        "bad anatomy",
        "Built the requested prompt",
        "illustration in moonlight",
        "a red dragon flies",
        "mountains surround the valley",
    )

    def test_single_mode_keeps_classic_path_and_empty_json_outputs(self):
        node = refiner.OnekoOllamaPromptRefiner()
        with mock.patch.object(node, "_refine_single", return_value=self.clean_result) as classic:
            result = node.refine(**refine_kwargs(pipeline_mode="single"))
        classic.assert_called_once()
        self.assertEqual(result[:2], self.clean_result[:2])
        self.assertEqual(result[3:6], self.clean_result[3:6])
        self.assertIn("original English source text was preserved", result[2])
        self.assertEqual(result[6:], ("{}", "{}"))

    def test_plan_compile_uses_planner_and_compiler_seed_contract(self):
        node = refiner.OnekoOllamaPromptRefiner()
        responses = [json.dumps(plan_data()), json.dumps(compiler_data())]
        with mock.patch.object(refiner, "_request_ollama", side_effect=responses) as request, mock.patch.object(
            refiner, "_postprocess_result", return_value=self.clean_result
        ):
            result = node.refine(**refine_kwargs())
        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args_list[0].args[3:5], (10, 0.2))
        self.assertIs(request.call_args_list[0].kwargs["output_schema"], refiner.PLAN_SCHEMA)
        self.assertEqual(request.call_args_list[1].args[3:5], (11, 0.45))
        self.assertIn("Structured plan from the planning stage", request.call_args_list[1].args[2])
        self.assertEqual(json.loads(result[6])["subject"], "a red dragon")
        self.assertEqual(result[7], "{}")

    def test_review_mode_runs_reviewer_without_correction_when_clean(self):
        node = refiner.OnekoOllamaPromptRefiner()
        responses = [json.dumps(plan_data()), json.dumps(compiler_data()), json.dumps(review_data())]
        with mock.patch.object(refiner, "_request_ollama", side_effect=responses) as request, mock.patch.object(
            refiner, "_postprocess_result", return_value=self.clean_result
        ):
            result = node.refine(**refine_kwargs(pipeline_mode="plan_compile_review"))
        self.assertEqual(request.call_count, 3)
        self.assertEqual(request.call_args_list[2].args[3:5], (12, 0.1))
        self.assertIs(request.call_args_list[2].kwargs["output_schema"], refiner.REVIEW_SCHEMA)
        review = json.loads(result[7])
        self.assertFalse(review["correction_applied"])
        self.assertEqual(review["final_local_issues"], [])

    def test_reviewer_can_trigger_exactly_one_correction(self):
        node = refiner.OnekoOllamaPromptRefiner()
        responses = [
            json.dumps(plan_data()),
            json.dumps(compiler_data()),
            json.dumps(review_data(needs_revision=True)),
            json.dumps(compiler_data()),
        ]
        with mock.patch.object(refiner, "_request_ollama", side_effect=responses) as request, mock.patch.object(
            refiner, "_postprocess_result", return_value=self.clean_result
        ):
            result = node.refine(**refine_kwargs(pipeline_mode="plan_compile_review"))
        self.assertEqual(request.call_count, 4)
        self.assertEqual(request.call_args_list[3].args[3:5], (13, 0.2))
        self.assertTrue(json.loads(result[7])["correction_applied"])

    def test_all_profiles_support_the_two_stage_pipeline(self):
        node = refiner.OnekoOllamaPromptRefiner()
        for profile in refiner.TARGET_PROFILES:
            negative = "" if profile == "z_image" else "bad anatomy"
            profile_result = (*self.clean_result[:1], negative, *self.clean_result[2:])
            with self.subTest(profile=profile), mock.patch.object(
                refiner,
                "_request_ollama",
                side_effect=[json.dumps(plan_data()), json.dumps(compiler_data())],
            ), mock.patch.object(refiner, "_postprocess_result", return_value=profile_result):
                result = node.refine(**refine_kwargs(profile=profile))
            self.assertEqual(len(result), 8)

    def test_reka_compiler_uses_schema_first_settings_for_verified_profiles(self):
        node = refiner.OnekoOllamaPromptRefiner()
        for profile in ("krea2", "z_image", "wan2_2_video", "pony_v7"):
            negative = "" if profile == "z_image" else "bad anatomy"
            profile_result = (*self.clean_result[:1], negative, *self.clean_result[2:])
            kwargs = refine_kwargs(profile=profile)
            kwargs["ollama_model"] = "autoren-reka-flash-3-21b-reasoning-q4:latest"
            with self.subTest(profile=profile), mock.patch.object(
                refiner,
                "_request_ollama",
                side_effect=[json.dumps(plan_data()), json.dumps(compiler_data())],
            ) as request, mock.patch.object(
                refiner, "_postprocess_result", return_value=profile_result
            ):
                node.refine(**kwargs)
            compiler_call = request.call_args_list[1]
            self.assertFalse(compiler_call.kwargs["reasoning"])
            self.assertEqual(compiler_call.kwargs["num_predict"], 900)

    def test_planner_failure_obeys_all_fallback_modes(self):
        node = refiner.OnekoOllamaPromptRefiner()
        with mock.patch.object(refiner, "_request_ollama", side_effect=RuntimeError("planner down")):
            with self.assertRaisesRegex(RuntimeError, "planner stage failed"):
                node.refine(**refine_kwargs(fallback_mode="strict"))

        with mock.patch.object(refiner, "_request_ollama", side_effect=ValueError("bad plan")), mock.patch.object(
            node, "_refine_single", return_value=self.clean_result
        ):
            adaptive = node.refine(**refine_kwargs(fallback_mode="adaptive"))
        self.assertIn("adaptive fallback to single after planner failure", adaptive[2])

        with mock.patch.object(
            refiner,
            "_request_ollama",
            side_effect=[ValueError("bad plan"), json.dumps(compiler_data())],
        ), mock.patch.object(refiner, "_postprocess_result", return_value=self.clean_result):
            continued = node.refine(**refine_kwargs(fallback_mode="continue"))
        continued_plan = json.loads(continued[6])
        self.assertEqual(continued_plan["planner_status"]["status"], "local_fallback")
        self.assertEqual(continued_plan["planner_status"]["stage_error"], "bad plan")
        self.assertIn("red dragon", continued_plan["subject"])
        self.assertIn("planner_local_fallback", continued[2])

    def test_continue_mode_replaces_repeated_foreign_subject_with_local_fallback(self):
        node = refiner.OnekoOllamaPromptRefiner()
        leaked = {
            "base_prompt": "anime illustration with soft rain and warm light",
            "foreground_prompt": (
                "A copper-haired archivist studies a blue crystal beside a brass machine."
            ),
            "background_prompt": (
                "A narrow wooden workshop surrounds her with rain-streaked windows."
            ),
            "negative": "bad anatomy",
            "report": "Built an Anima prompt.",
        }
        kwargs = refine_kwargs(
            profile="anima",
            pipeline_mode="plan_compile",
            fallback_mode="continue",
        )
        kwargs["word_salad"] = "maid latex tentacles forest fearful wet eyes"
        kwargs["style_anchor"] = "anime style"
        with mock.patch.object(
            refiner,
            "_request_ollama",
            side_effect=[
                ValueError("planner rejected"),
                json.dumps(leaked),
                json.dumps(leaked),
            ],
        ):
            result = node.refine(**kwargs)
        self.assertNotIn("copper-haired archivist", result[0].lower())
        self.assertNotIn("blue crystal", result[0].lower())
        self.assertIn("maid", result[0].lower())
        self.assertIn("final_local_fallback", result[2])
        plan = json.loads(result[6])
        self.assertEqual(plan["planner_status"]["status"], "local_fallback")

    def test_compiler_reviewer_and_correction_failures_are_bounded(self):
        node = refiner.OnekoOllamaPromptRefiner()
        with mock.patch.object(
            refiner,
            "_request_ollama",
            side_effect=[json.dumps(plan_data()), ValueError("bad compiler")],
        ):
            with self.assertRaisesRegex(RuntimeError, "compiler stage failed"):
                node.refine(**refine_kwargs(fallback_mode="strict"))

        with mock.patch.object(
            refiner,
            "_request_ollama",
            side_effect=[json.dumps(plan_data()), json.dumps(compiler_data()), ValueError("bad review")],
        ), mock.patch.object(refiner, "_postprocess_result", return_value=self.clean_result):
            review_continue = node.refine(
                **refine_kwargs(pipeline_mode="plan_compile_review", fallback_mode="continue")
            )
        self.assertIn("bad review", json.loads(review_continue[7])["stage_error"])

        required_plan = plan_data(required=["spreadsheet"])
        with mock.patch.object(
            refiner,
            "_request_ollama",
            side_effect=[json.dumps(required_plan), json.dumps(compiler_data()), ValueError("bad correction")],
        ), mock.patch.object(refiner, "_postprocess_result", return_value=self.clean_result) as postprocess:
            correction_continue = node.refine(**refine_kwargs(fallback_mode="continue"))
        self.assertEqual(postprocess.call_count, 2)
        self.assertIn("correction_kept_compiler", correction_continue[2])

    def test_pipeline_and_fallback_mode_participate_in_cache_hash(self):
        base = refine_kwargs(profile="pony_v6", pipeline_mode="single", fallback_mode="strict")
        single_strict = refiner.OnekoOllamaPromptRefiner.IS_CHANGED(**base)
        base["fallback_mode"] = "continue"
        single_continue = refiner.OnekoOllamaPromptRefiner.IS_CHANGED(**base)
        self.assertEqual(single_strict, single_continue)
        base["pipeline_mode"] = "plan_compile"
        chain_continue = refiner.OnekoOllamaPromptRefiner.IS_CHANGED(**base)
        base["fallback_mode"] = "strict"
        chain_strict = refiner.OnekoOllamaPromptRefiner.IS_CHANGED(**base)
        self.assertNotEqual(chain_continue, chain_strict)
        self.assertNotEqual(single_continue, chain_continue)


if __name__ == "__main__":
    unittest.main()
