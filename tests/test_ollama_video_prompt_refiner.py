import json
import sys
import unittest
from pathlib import Path
from unittest import mock


from tests.paths import COMFY_ROOT, ONEKO_ROOT
CUSTOM_NODE_ROOT = ONEKO_ROOT
for import_path in (COMFY_ROOT, CUSTOM_NODE_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from Oneko_ComfyUI_Nodes.nodes import ollama_video_prompt_refiner as video


def _long_section(opening, target=84):
    words = opening.split()
    filler = (
        "specific grounded cinematic detail supports coherent motion stable continuity natural timing "
        "careful composition physical response atmospheric depth readable staging polished direction"
    ).split()
    while video._word_count(" ".join(words)) < target:
        words.append(filler[len(words) % len(filler)])
    return " ".join(words)


def _values(**updates):
    values = {
        "scene": _long_section("A glowing forest remains stable at twilight."),
        "character": _long_section("A young blue-haired mage holds a glowing orb."),
        "action": _long_section('She turns toward the camera and says "Hallo, ist da jemand?"'),
        "camera": _long_section("A medium shot slowly pushes forward."),
        "visual_style": _long_section("Clean anime line art uses soft cinematic light."),
        "audio": _long_section('A clear young female voice says "Hallo, ist da jemand?" over quiet forest ambience.'),
        "negative": "flicker, watermark",
        "report": "Kept the requested mage and continuous camera move.",
    }
    values.update(updates)
    return values


def _kwargs(**updates):
    values = {
        "scene": "A glowing forest at twilight.",
        "character": "A young mage with long blue hair.",
        "action": 'She turns toward the camera and says "Hallo, ist da jemand?"',
        "camera": "A medium shot with a slow forward move.",
        "visual_style": "Beautiful anime style with soft cinematic light.",
        "audio": 'A young female voice says "Hallo, ist da jemand?" over quiet forest ambience.',
        "ollama_url": "http://127.0.0.1:11434",
        "ollama_model": "test-model",
        "target_profile": "minimax_h3",
        "seed": 7,
        "temperature": 0.45,
        "top_p": 0.9,
        "timeout_seconds": 30,
        "context_length": "4096",
        "creativity_mode": "balanced",
        "pipeline_mode": "single",
        "fallback_mode": "adaptive",
        "unload_after_run": True,
    }
    values.update(updates)
    return values


def _german_kwargs(**updates):
    values = _kwargs(
        scene="Ein leuchtender Wald in der Dämmerung.",
        character="Eine junge Magierin mit langen blauen Haaren.",
        action='Sie dreht sich zur Kamera und sagt "Hallo, ist da jemand?"',
        camera="Die Kamera bewegt sich langsam nach vorne.",
        visual_style="Schöner Anime Stil mit weichem Licht.",
        audio='Eine junge Stimme sagt "Hallo, ist da jemand?" über ruhiger Waldatmosphäre.',
    )
    values.update(updates)
    return values


def _language_values(**updates):
    values = {key: _kwargs()[key] for key in video.VIDEO_SECTION_KEYS}
    values.update(updates)
    return values


def _response(values=None):
    return json.dumps(values or _values(), ensure_ascii=False)


def _review(**updates):
    values = {
        "all_required_preserved": True,
        "single_shot_consistent": True,
        "camera_consistent": True,
        "quotes_preserved": True,
        "needs_correction": False,
        "issues": [],
        "summary": "The prompt is grounded and internally consistent.",
    }
    values.update(updates)
    return json.dumps(values)


class OllamaVideoPromptRefinerTests(unittest.TestCase):
    def setUp(self):
        self.node = video.OnekoOllamaVideoPromptRefiner()
        self.unload_patch = mock.patch.object(video, "_unload_after_run")
        self.unload = self.unload_patch.start()
        self.translation_patch = mock.patch.object(
            self.node,
            "_translate_source",
            side_effect=lambda source, *_args, **_kwargs: (source, "language_test_bypass"),
        )
        self.translation = self.translation_patch.start()
        self.translation_active = True

    def tearDown(self):
        if self.translation_active:
            self.translation_patch.stop()
        self.unload_patch.stop()

    def use_real_translation_stage(self):
        if self.translation_active:
            self.translation_patch.stop()
            self.translation_active = False

    def test_registration_interface_and_model_dropdown(self):
        self.assertIs(
            video.NODE_CLASS_MAPPINGS["OnekoOllamaVideoPromptRefiner"],
            video.OnekoOllamaVideoPromptRefiner,
        )
        self.assertEqual(
            video.NODE_DISPLAY_NAME_MAPPINGS["OnekoOllamaVideoPromptRefiner"],
            "Ollama Video Prompt Refiner (Oneko)",
        )
        self.assertEqual(video.OnekoOllamaVideoPromptRefiner.CATEGORY, "Oneko/02 Text/Ollama")
        self.assertEqual(video.OnekoOllamaVideoPromptRefiner.RETURN_NAMES, ("prompt", "negative", "report"))

        with mock.patch.object(video, "_available_ollama_models", return_value=["model-a", "model-b"]):
            inputs = video.OnekoOllamaVideoPromptRefiner.INPUT_TYPES()
        for key in video.VIDEO_SECTION_KEYS:
            self.assertTrue(inputs["required"][key][1]["multiline"])
            self.assertTrue(inputs["required"][key][1]["defaultInput"])
        self.assertEqual(inputs["required"]["target_profile"][0], video.VIDEO_TARGET_PROFILES)
        self.assertEqual(inputs["required"]["ollama_model"][0], ["model-a", "model-b"])
        self.assertEqual(inputs["optional"]["creativity_mode"][0], ("faithful", "balanced", "cinematic"))
        self.assertEqual(inputs["optional"]["creativity_mode"][1]["default"], "balanced")
        self.assertEqual(inputs["optional"]["pipeline_mode"][0], ("single", "review"))

        frontend = (CUSTOM_NODE_ROOT / "web" / "ollama_model_select.js").read_text(encoding="utf-8")
        self.assertIn("OnekoOllamaVideoPromptRefiner", frontend)

    def test_minimax_prompt_has_fixed_header_order_and_negative_baseline(self):
        with mock.patch.object(video, "_request_ollama", return_value=_response()):
            prompt, negative, report = self.node.refine(**_kwargs())

        headers = ["[Scene]", "[Character]", "[Action]", "[Camera]", "[Visual Style]", "[Audio]"]
        positions = [prompt.index(header) for header in headers]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(prompt.count('"Hallo, ist da jemand?"'), 2)
        for term in ("flicker", "temporal jitter", "identity drift", "lip-sync mismatch", "unwanted dialogue"):
            self.assertIn(term, negative)
        self.assertIn("profile=minimax_h3", report)

    def test_short_or_empty_minimax_sections_trigger_repair(self):
        values = _values(character="", camera="", visual_style="", audio="", action="A leaf falls.")
        kwargs = _kwargs(action="A leaf falls.", audio="")
        with mock.patch.object(video, "_request_ollama", side_effect=[_response(values), _response()]) as request:
            prompt, _negative, _report = self.node.refine(**kwargs)

        self.assertEqual(request.call_count, 2)
        for header in ("[Scene]", "[Character]", "[Action]", "[Camera]", "[Visual Style]", "[Audio]"):
            self.assertIn(header, prompt)

    def test_wan_prompt_order_excludes_audio_and_keeps_baseline(self):
        values = _values(
            character="CHARACTER PART.",
            action="ACTION PART.",
            scene="SCENE PART.",
            camera="CAMERA PART.",
            visual_style="STYLE PART.",
            audio="AUDIO MUST NOT APPEAR.",
        )
        kwargs = _kwargs(
            target_profile="wan2_2_video",
            action="One continuous turn.",
            audio="Ignored music.",
        )
        with mock.patch.object(video, "_request_ollama", return_value=_response(values)):
            prompt, negative, report = self.node.refine(**kwargs)

        expected = "CHARACTER PART. ACTION PART. SCENE PART. CAMERA PART. STYLE PART."
        self.assertEqual(prompt, expected)
        self.assertNotIn("AUDIO MUST NOT APPEAR", prompt)
        self.assertIn("Audio was excluded", report)
        for term in ("flicker", "temporal jitter", "identity drift", "frozen motion", "camera shake"):
            self.assertIn(term, negative)

    def test_german_prose_can_be_translated_while_dialogue_stays_exact(self):
        self.use_real_translation_stage()
        with mock.patch.object(
            video,
            "_request_ollama",
            side_effect=[json.dumps(_language_values(), ensure_ascii=False), _response()],
        ) as request:
            prompt, _negative, report = self.node.refine(**_german_kwargs())

        self.assertEqual(request.call_count, 2)
        self.assertIn("A glowing forest remains stable at twilight", prompt)
        self.assertNotIn("Leuchtender Wald", prompt)
        self.assertEqual(prompt.count('"Hallo, ist da jemand?"'), 2)
        self.assertIn("language_translated", report)

    def test_translation_stage_also_runs_for_english_input(self):
        self.use_real_translation_stage()
        with mock.patch.object(
            video,
            "_request_ollama",
            side_effect=[json.dumps(_language_values()), _response()],
        ) as request:
            _prompt, _negative, report = self.node.refine(**_kwargs())

        self.assertEqual(request.call_count, 2)
        self.assertIn("language_translated", report)

    def test_untranslated_language_stage_triggers_its_own_repair(self):
        self.use_real_translation_stage()
        untranslated = {key: _german_kwargs()[key] for key in video.VIDEO_SECTION_KEYS}
        with mock.patch.object(
            video,
            "_request_ollama",
            side_effect=[
                json.dumps(untranslated, ensure_ascii=False),
                json.dumps(_language_values(), ensure_ascii=False),
                _response(),
            ],
        ) as request:
            prompt, _negative, report = self.node.refine(**_german_kwargs())

        self.assertEqual(request.call_count, 3)
        self.assertNotIn("Ein leuchtender Wald", prompt)
        self.assertEqual(prompt.count('"Hallo, ist da jemand?"'), 2)
        self.assertIn("language_repair", report)

    def test_untranslated_compiler_output_triggers_compiler_repair(self):
        german_output = _values(
            scene=_long_section("Ein leuchtender Wald bleibt in der Dämmerung stabil."),
            character=_long_section("Eine junge Magierin mit langen blauen Haaren hält eine leuchtende Kugel."),
            action=_long_section('Sie dreht sich zur Kamera und sagt "Hallo, ist da jemand?"'),
            camera=_long_section("Die Kamera bewegt sich langsam nach vorne und bleibt stabil."),
            visual_style=_long_section("Schöner Anime Stil mit weichem Licht und klaren Linien."),
            audio=_long_section('Eine junge Stimme sagt "Hallo, ist da jemand?" über ruhiger Waldatmosphäre.'),
        )
        with mock.patch.object(
            video,
            "_request_ollama",
            side_effect=[_response(german_output), _response()],
        ) as request:
            prompt, _negative, report = self.node.refine(**_kwargs())

        self.assertEqual(request.call_count, 2)
        self.assertNotIn("Ein leuchtender Wald", prompt)
        self.assertIn("compiler_repair", report)

    def test_whitespace_normalization_does_not_change_quoted_dialogue(self):
        self.use_real_translation_stage()
        dialogue = '"Bleib  genau hier!"'
        kwargs = _kwargs(
            action=f"Sie ruft {dialogue}",
            audio=f"Eine Stimme ruft {dialogue}",
        )
        values = _values(
            action=f"{_long_section('She calls out')} {dialogue}",
            audio=f"{_long_section('A voice calls out')} {dialogue}",
        )
        language_values = _language_values(
            action=f"She calls out {dialogue}",
            audio=f"A voice calls out {dialogue}",
        )
        with mock.patch.object(
            video,
            "_request_ollama",
            side_effect=[json.dumps(language_values, ensure_ascii=False), _response(values)],
        ):
            prompt, _negative, _report = self.node.refine(**kwargs)

        self.assertEqual(prompt.count(dialogue), 2)

    def test_multiple_prequoted_speaker_lines_are_tracked_individually(self):
        dialogue = 'Frau: "nein, bitte nicht!" danach Mann: "sei still!!"'

        self.assertEqual(
            video._quoted_tokens(dialogue),
            ['"nein, bitte nicht!"', '"sei still!!"'],
        )

    def test_language_stage_restores_dropped_multi_speaker_lines_locally(self):
        dialogue = 'Frau: "nein, bitte nicht!" danach Mann: "sei still!!"'
        source = {key: _kwargs()[key] for key in video.VIDEO_SECTION_KEYS}
        source["action"] = f"Sie spricht. {dialogue}"
        source["audio"] = f"Zwei Stimmen sprechen. {dialogue}"
        translated = _language_values(
            action="She speaks without returning the quoted lines.",
            audio="Two voices speak without returning the quoted lines.",
        )

        result = video._validate_language_result(translated, source)

        for token in ('"nein, bitte nicht!"', '"sei still!!"'):
            self.assertIn(token, result["action"])
            self.assertIn(token, result["audio"])

    def test_compiler_restores_multi_speaker_lines_above_h3_minimum(self):
        dialogue = 'Frau: "nein, bitte nicht!" danach Mann: "sei still!!"'
        source = {key: _kwargs()[key] for key in video.VIDEO_SECTION_KEYS}
        source["action"] = f"Sie spricht. {dialogue}"
        source["audio"] = f"Zwei Stimmen sprechen. {dialogue}"
        values = _values(
            action=_long_section("She speaks while the continuous action remains grounded.", target=96),
            audio=_long_section("Two voices remain clear over grounded room ambience.", target=96),
        )

        result = video._validate_video_result(values, source, "minimax_h3")

        for key in ("action", "audio"):
            self.assertGreaterEqual(video._word_count(result[key]), 80)
            for token in ('"nein, bitte nicht!"', '"sei still!!"'):
                self.assertIn(token, result[key])

    def test_compiler_restores_single_dialogue_and_removes_invented_quoted_lines(self):
        source = {key: _kwargs()[key] for key in video.VIDEO_SECTION_KEYS}
        values = _values(
            action=_long_section(
                "She completes the requested motion and then says 'translated line' while another voice adds 'invented reply'."
            ),
            audio=_long_section(
                "The ambience remains coherent while a voice says 'translated line' and another adds 'invented reply'."
            ),
        )

        result = video._validate_video_result(values, source, "minimax_h3")

        for key in ("action", "audio"):
            self.assertIn('"Hallo, ist da jemand?"', result[key])
            self.assertNotIn("'translated line'", result[key])
            self.assertNotIn("'invented reply'", result[key])

    def test_changed_or_missing_dialogue_triggers_one_repair(self):
        invalid = _values(audio="A voice says something else.")
        with mock.patch.object(video, "_request_ollama", side_effect=[_response(invalid), _response()]) as request:
            prompt, _negative, report = self.node.refine(**_kwargs())

        self.assertEqual(request.call_count, 2)
        self.assertIn('"Hallo, ist da jemand?"', prompt)
        self.assertIn("compiler_repair", report)

    def test_invalid_json_is_repaired_exactly_once(self):
        with mock.patch.object(video, "_request_ollama", side_effect=["not json", _response()]) as request:
            self.node.refine(**_kwargs())

        self.assertEqual(request.call_count, 2)

    def test_review_pass_does_not_request_correction(self):
        kwargs = _kwargs(pipeline_mode="review")
        with mock.patch.object(video, "_request_ollama", side_effect=[_response(), _review()]) as request:
            _prompt, _negative, report = self.node.refine(**kwargs)

        self.assertEqual(request.call_count, 2)
        self.assertIn("review_passed", report)

    def test_review_findings_request_at_most_one_correction(self):
        kwargs = _kwargs(pipeline_mode="review")
        review = _review(
            all_required_preserved=False,
            needs_correction=True,
            issues=["The requested orb was omitted."],
            summary="Restore the orb.",
        )
        corrected = _values(report="Restored the requested orb.")
        with mock.patch.object(
            video,
            "_request_ollama",
            side_effect=[_response(), review, _response(corrected)],
        ) as request:
            _prompt, _negative, report = self.node.refine(**kwargs)

        self.assertEqual(request.call_count, 3)
        self.assertIn("review_corrected", report)

    def test_review_failure_modes(self):
        adaptive = _kwargs(pipeline_mode="review", fallback_mode="adaptive")
        with mock.patch.object(video, "_request_ollama", side_effect=[_response(), RuntimeError("review offline")]):
            _prompt, _negative, report = self.node.refine(**adaptive)
        self.assertIn("review_skipped", report)

        strict = _kwargs(pipeline_mode="review", fallback_mode="strict")
        with mock.patch.object(video, "_request_ollama", side_effect=[_response(), RuntimeError("review offline")]):
            with self.assertRaisesRegex(RuntimeError, "review failed"):
                self.node.refine(**strict)

    def test_adaptive_uses_local_fallback_after_two_invalid_responses(self):
        with mock.patch.object(video, "_request_ollama", side_effect=["bad", "still bad"]) as request:
            prompt, negative, report = self.node.refine(**_kwargs(fallback_mode="adaptive"))

        self.assertEqual(request.call_count, 2)
        self.assertIn("[Scene]\nA glowing forest", prompt)
        self.assertIn('"Hallo, ist da jemand?"', prompt)
        self.assertIn("temporal jitter", negative)
        self.assertIn("compiler_validation_fallback", report)

    def test_adaptive_keeps_creative_json_when_only_section_length_is_short(self):
        initial = _values(scene="A cinematic river temple glows through evening rain.")
        repaired = _values(scene="Layered candlelight ripples across a cinematic river temple in evening rain.")
        with mock.patch.object(
            video,
            "_request_ollama",
            side_effect=[_response(initial), _response(repaired)],
        ) as request:
            prompt, _negative, report = self.node.refine(**_kwargs(fallback_mode="adaptive"))

        self.assertEqual(request.call_count, 2)
        self.assertIn("Layered candlelight ripples", prompt)
        self.assertNotIn("[Scene]\nA glowing forest at twilight.", prompt)
        self.assertIn("compiler_relaxed_fallback", report)

    def test_strict_rejects_two_invalid_responses(self):
        with mock.patch.object(video, "_request_ollama", side_effect=["bad", "still bad"]):
            with self.assertRaisesRegex(RuntimeError, "invalid twice"):
                self.node.refine(**_kwargs(fallback_mode="strict"))

    def test_continue_handles_transport_failure_but_adaptive_does_not(self):
        with mock.patch.object(video, "_request_ollama", side_effect=RuntimeError("offline")):
            prompt, _negative, report = self.node.refine(**_kwargs(fallback_mode="continue"))
        self.assertIn("[Scene]", prompt)
        self.assertIn("compiler_transport_fallback", report)

        with mock.patch.object(video, "_request_ollama", side_effect=RuntimeError("offline")):
            with self.assertRaisesRegex(RuntimeError, "offline"):
                self.node.refine(**_kwargs(fallback_mode="adaptive"))

    def test_unload_runs_after_success_and_runtime_error(self):
        with mock.patch.object(video, "_request_ollama", return_value=_response()):
            self.node.refine(**_kwargs())
        self.unload.assert_called_once()

        self.unload.reset_mock()
        with mock.patch.object(video, "_request_ollama", side_effect=RuntimeError("offline")):
            with self.assertRaises(RuntimeError):
                self.node.refine(**_kwargs(fallback_mode="strict"))
        self.unload.assert_called_once()

    def test_empty_source_is_rejected_without_calling_ollama(self):
        kwargs = _kwargs(**{key: "" for key in video.VIDEO_SECTION_KEYS})
        with mock.patch.object(video, "_request_ollama") as request:
            with self.assertRaisesRegex(RuntimeError, "at least one video section"):
                self.node.refine(**kwargs)
        request.assert_not_called()

    def test_is_changed_covers_public_inputs(self):
        base = _kwargs()
        digest = video.OnekoOllamaVideoPromptRefiner.IS_CHANGED(**base)
        changes = {
            "scene": "different scene",
            "audio": "different audio",
            "ollama_url": "http://example.test:11434",
            "ollama_model": "different-model",
            "target_profile": "wan2_2_video",
            "seed": 8,
            "temperature": 0.2,
            "top_p": 0.8,
            "timeout_seconds": 40,
            "context_length": "8192",
            "creativity_mode": "cinematic",
            "pipeline_mode": "review",
            "fallback_mode": "strict",
            "unload_after_run": False,
        }
        for field, value in changes.items():
            changed = dict(base)
            changed[field] = value
            self.assertNotEqual(
                digest,
                video.OnekoOllamaVideoPromptRefiner.IS_CHANGED(**changed),
                field,
            )

    def test_compiler_rules_forbid_unrelated_main_subjects(self):
        instructions = video.VIDEO_SYSTEM_INSTRUCTIONS.lower()
        self.assertIn("never invent a new main character", instructions)
        self.assertIn("add unrelated objects", instructions)
        self.assertIn("preserve every double-quoted string exactly", instructions)

    def test_balanced_and_cinematic_modes_require_real_enrichment(self):
        balanced = video._creativity_instructions("balanced").lower()
        cinematic = video._creativity_instructions("cinematic").lower()
        self.assertIn("do not copy or merely join", balanced)
        self.assertIn("production-ready", balanced)
        self.assertIn("experienced film director", cinematic)
        self.assertIn("sound designer", cinematic)

    def test_minimax_accepts_sixty_words_without_upper_limit(self):
        source = {key: _kwargs()[key] for key in video.VIDEO_SECTION_KEYS}
        valid = _values()
        validated = video._validate_video_result(valid, source, "minimax_h3")
        for key in video.VIDEO_SECTION_KEYS:
            self.assertGreaterEqual(video._word_count(validated[key]), 60)

        too_short = _values(scene=" ".join(["detail"] * 59))
        with self.assertRaisesRegex(ValueError, "scene=59"):
            video._validate_video_result(too_short, source, "minimax_h3")

        minimum_result = video._validate_video_result(
            _values(scene=" ".join(["detail"] * 60)),
            source,
            "minimax_h3",
        )
        self.assertEqual(video._word_count(minimum_result["scene"]), 60)

        long_result = video._validate_video_result(
            _values(scene=" ".join(["detail"] * 250)),
            source,
            "minimax_h3",
        )
        self.assertEqual(video._word_count(long_result["scene"]), 250)

    def test_minimax_prompt_requests_one_hundred_words_per_section(self):
        instructions = video._profile_instructions("minimax_h3")

        self.assertIn("approximately 100 words", instructions)
        self.assertIn("90 to 120 words", instructions)
        self.assertIn("Never exceed 140 words", instructions)
        self.assertIn("avoid padding, repetition, or runaway verbosity", instructions)

    def test_compiler_disables_thinking_and_has_sufficient_json_budget(self):
        with mock.patch.object(video, "_request_ollama", return_value=_response()) as request:
            self.node.refine(**_kwargs())

        compiler_call = request.call_args
        self.assertFalse(compiler_call.kwargs["reasoning"])
        self.assertEqual(
            compiler_call.kwargs["num_predict"],
            video.VIDEO_COMPILER_NUM_PREDICT,
        )
        self.assertGreaterEqual(video.VIDEO_COMPILER_NUM_PREDICT, 2400)


if __name__ == "__main__":
    unittest.main()
