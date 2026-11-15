import json
import sys
import unittest
from unittest import mock

from tests.paths import COMFY_ROOT

if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

from Oneko_ComfyUI_Nodes.nodes import ollama_prompt_refiner as refiner


LABEL = '"Grüße: Café_7 — 20%"'
SOURCE = (
    'One red fox stands beside three blue glass bottles. '
    f'A brass sign reads {LABEL}. '
    'Two white birds perch to the left of a wooden arch behind the fox.'
)
ANCHOR = 'Soft watercolor illustration'


def values(**overrides):
    result = {
        'base_prompt': 'Soft watercolor illustration with gentle morning light.',
        'foreground_prompt': (
            'One red fox stands beside three blue glass bottles. '
            f'A brass sign reads {LABEL}.'
        ),
        'background_prompt': 'Two white birds perch to the left of a wooden arch behind the fox.',
        'negative': '',
        'report': 'Preserved the fox, counted objects, positions, and sign lettering.',
    }
    result.update(overrides)
    return result


def refine_kwargs(**overrides):
    result = {
        'word_salad': SOURCE,
        'ollama_url': 'http://127.0.0.1:11434',
        'ollama_model': 'test-model',
        'target_profile': 'moss_image',
        'seed': 31,
        'temperature': 0.4,
        'top_p': 0.9,
        'style_cluster': 430,
        'timeout_seconds': 30,
        'context_length': 8192,
        'style_anchor': ANCHOR,
        'fallback_mode': 'adaptive',
        'pipeline_mode': 'single',
        'unload_after_run': False,
    }
    result.update(overrides)
    return result


def passthrough_language(word_salad, left, right, top, bottom, *_args):
    return refiner._language_source_values(word_salad, left, right, top, bottom), 'english', False


class MossPromptProfileTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(refiner, '_prepare_language_inputs', side_effect=passthrough_language)
        patcher.start()
        self.addCleanup(patcher.stop)

    def postprocess(self, response=None, source=SOURCE, anchor=ANCHOR, mode='adaptive'):
        return refiner._postprocess_result(
            response if response is not None else values(),
            'moss_image', source, anchor, 430, fallback_mode=mode,
        )

    def test_profile_is_appended_and_existing_interface_contract_is_unchanged(self):
        old_profiles = ('pony_v6', 'illustrious', 'pony_v7', 'z_image', 'anima', 'wan2_2_video', 'krea2')
        self.assertEqual(refiner.TARGET_PROFILES, (*old_profiles, 'moss_image'))
        with mock.patch.object(refiner, '_available_ollama_models', return_value=['test-model']):
            profile, options = refiner.OnekoOllamaPromptRefiner.INPUT_TYPES()['required']['target_profile']
        self.assertEqual(profile, refiner.TARGET_PROFILES)
        self.assertEqual(options['default'], 'pony_v7')
        self.assertEqual(refiner._normalize_target_profile('moss_image'), 'moss_image')
        self.assertEqual(refiner.OnekoOllamaPromptRefiner.RETURN_NAMES,
                         ('positive', 'negative', 'report', 'base_prompt', 'foreground_prompt',
                          'background_prompt', 'plan_json', 'review_json'))
        self.assertEqual(refiner.OnekoOllamaPromptRefiner.RETURN_TYPES, ('STRING',) * 8)

    def test_moss_generation_has_no_foreign_word_budget_or_sample_scene(self):
        prompt = refiner._build_generation_prompt('moss_image', SOURCE, ANCHOR, 430)
        self.assertIn('MOSS', prompt)
        self.assertIn(SOURCE, prompt)
        self.assertIn(ANCHOR, prompt)
        for foreign_budget in ('300 to 360', '140 to 165', '105 to 125'):
            self.assertNotIn(foreign_budget, prompt)
        for motif in ('copper-haired archivist', 'narrow wooden workshop', 'rain-streaked window'):
            self.assertNotIn(motif, prompt.lower())
        self.assertEqual(refiner._candidate_few_shot_example('moss_image', SOURCE, ANCHOR), '')

    def test_subject_first_result_preserves_counts_positions_anchor_and_visible_text(self):
        positive, negative, _report, base, foreground, background = self.postprocess()
        self.assertTrue(positive.startswith(foreground))
        self.assertLess(positive.index(foreground), positive.index(background))
        self.assertTrue(base.startswith(ANCHOR))
        self.assertIn('three blue glass bottles', positive)
        self.assertIn('Two white birds', positive)
        self.assertIn('to the left of a wooden arch behind the fox', positive)
        self.assertIn(LABEL, positive)
        self.assertEqual(negative, '')
        self.assertEqual(foreground, values()['foreground_prompt'])
        self.assertEqual(background, values()['background_prompt'])

    def test_short_natural_sections_are_not_padded(self):
        response = values(
            base_prompt='Soft watercolor illustration.',
            foreground_prompt='A red fox watches three blue bottles.',
            background_prompt='Two white birds perch behind the fox.',
        )
        result = self.postprocess(response, source='A red fox watches three blue bottles. Two white birds perch behind the fox.')
        self.assertEqual(result[4], response['foreground_prompt'])
        self.assertEqual(result[5], response['background_prompt'])
        self.assertLess(refiner._word_count(result[0]), 40)

    def test_long_valid_content_is_not_cut_at_krea2_word_limits(self):
        foreground = ' '.join(f'On shelf {index}, three blue glass bottles reflect soft daylight.' for index in range(1, 42))
        foreground += f' A brass sign reads {LABEL}.'
        response = values(foreground_prompt=foreground)
        result = self.postprocess(response, source=foreground + ' ' + response['background_prompt'])
        self.assertGreater(refiner._word_count(result[0]), 360)
        self.assertEqual(result[4], foreground)
        self.assertIn('On shelf 41,', result[0])
        self.assertIn(LABEL, result[0])

    def test_quoted_control_like_text_and_exact_whitespace_survive_cleaning(self):
        label = '"score_9, Pony v7:  OPEN!\nCafé_7 — 20%"'
        foreground = f'A red fox holds a brass sign that reads {label}.'
        response = values(foreground_prompt=foreground)
        result = self.postprocess(response, source=foreground)
        self.assertIn(label, result[4])
        self.assertIn(label, result[0])

    def test_user_style_anchor_is_preserved_exactly(self):
        anchor = 'MyStyle, (ink:1.2), <lora:custom:0.8>'
        result = self.postprocess(values(base_prompt='Gentle morning light creates a readable composition.'), anchor=anchor)
        self.assertTrue(result[3].startswith(anchor))
        self.assertIn(anchor, result[0])

    def test_complete_anchor_with_changed_case_and_final_punctuation_is_not_duplicated(self):
        anchor = 'Soft watercolor illustration with restrained warm colors.'
        response = values(base_prompt='soft watercolor illustration with restrained warm colors')
        result = self.postprocess(response, anchor=anchor)
        self.assertEqual(result[3], anchor)
        self.assertEqual(result[0].casefold().count('soft watercolor illustration with restrained warm colors'), 1)
        self.assertIn(anchor, result[0])

    def test_negative_is_optional_and_nonempty_negative_remains_available(self):
        empty = self.postprocess(values(negative=''))
        supplied = self.postprocess(values(negative='blurry fur, duplicate objects'))
        self.assertEqual(empty[1], '')
        self.assertEqual(supplied[1], 'blurry fur, duplicate objects')

    def test_missing_or_changed_visible_label_is_rejected_for_repair(self):
        bad = values(foreground_prompt='One red fox stands beside three blue bottles. A brass sign reads "Cafe 7".')
        with self.assertRaises(ValueError):
            self.postprocess(bad)

    def test_visible_text_substring_is_not_accepted_as_exact_lettering(self):
        source = 'A red fox holds a brass sign reading "OPEN".'
        bad = values(foreground_prompt='A red fox holds a brass sign reading "REOPENED".')
        with self.assertRaisesRegex(ValueError, 'visible text'):
            self.postprocess(bad, source=source)

    def test_visible_text_in_report_or_negative_does_not_replace_positive_lettering(self):
        source = 'A red fox holds a brass sign reading "OPEN".'
        bad = values(
            foreground_prompt='A red fox holds a blank brass sign beside a quiet pool.',
            negative='blurred lettering, "OPEN"',
            report='Preserved the requested label "OPEN".',
        )
        with self.assertRaisesRegex(ValueError, 'visible text'):
            self.postprocess(bad, source=source)

    def test_quoted_style_instruction_is_not_treated_as_required_image_lettering(self):
        source = 'A red fox beside three blue bottles. The requested style is "storybook watercolor".'
        response = values(
            base_prompt='A storybook watercolor uses gentle morning light.',
            foreground_prompt='One red fox stands beside three blue bottles.',
            background_prompt='Two white birds perch behind the fox.',
        )
        result = self.postprocess(response, source=source, anchor='')
        self.assertIn('storybook watercolor', result[0])
        self.assertNotIn('"storybook watercolor"', result[0])

    def test_valid_node_response_preserves_output_contract_and_reports_initial_stage(self):
        with mock.patch.object(refiner, '_request_ollama', return_value=json.dumps(values(), ensure_ascii=False)) as request:
            result = refiner.OnekoOllamaPromptRefiner().refine(**refine_kwargs())
        self.assertEqual(request.call_count, 1)
        self.assertEqual(len(result), 8)
        self.assertIn(LABEL, result[0])
        self.assertIn('stage=initial', result[2])
        self.assertIn('fallback_mode=adaptive', result[2])
        self.assertEqual(result[6:], ('{}', '{}'))

    def test_repair_restores_exact_visible_text_and_uses_incremented_seed(self):
        bad = values(foreground_prompt='One red fox stands beside three blue bottles. A brass sign reads "Cafe 7".')
        replies = [json.dumps(bad), json.dumps(values(), ensure_ascii=False)]
        with mock.patch.object(refiner, '_request_ollama', side_effect=replies) as request:
            result = refiner.OnekoOllamaPromptRefiner().refine(**refine_kwargs())
        self.assertEqual(request.call_count, 2)
        self.assertEqual([call.args[3] for call in request.call_args_list], [31, 32])
        self.assertIn(SOURCE, request.call_args_list[1].args[2])
        self.assertIn(LABEL, result[0])
        self.assertNotIn('"Cafe 7"', result[0])
        self.assertIn('stage=repair', result[2])

    def test_minimal_retry_keeps_source_regions_and_literal_text(self):
        region = 'three blue lanterns beside a sign reading "NORTH_3"'
        response = values(background_prompt='On the left, three blue lanterns stand beside a sign reading "NORTH_3". ' + values()['background_prompt'])
        with mock.patch.object(refiner, '_request_ollama', side_effect=['invalid', 'still invalid', json.dumps(response, ensure_ascii=False)]) as request:
            result = refiner.OnekoOllamaPromptRefiner().refine(**refine_kwargs(left=region))
        self.assertEqual(request.call_count, 3)
        self.assertEqual([call.args[3] for call in request.call_args_list], [31, 32, 33])
        for call in request.call_args_list:
            self.assertIn(SOURCE, call.args[2])
            self.assertIn(region, call.args[2])
        self.assertIn(LABEL, result[0])
        self.assertIn('"NORTH_3"', result[0])
        self.assertIn('stage=minimal', result[2])

    def test_adaptive_invalid_responses_fall_back_to_full_source(self):
        source = SOURCE + ' Exactly four red lanterns hang above seven flat stones at the far right.'
        with mock.patch.object(refiner, '_request_ollama', side_effect=['invalid', 'invalid again', 'still invalid']) as request:
            result = refiner.OnekoOllamaPromptRefiner().refine(**refine_kwargs(word_salad=source))
        self.assertEqual(request.call_count, 3)
        self.assertIn(source, result[0])
        self.assertIn(LABEL, result[0])
        self.assertIn('Exactly four red lanterns hang above seven flat stones at the far right.', result[0])
        self.assertEqual(result[1], '')
        self.assertIn('stage=local', result[2])
        for invented_scene in ('crystal tower', 'narrow wooden workshop', 'coherent anime subject'):
            self.assertNotIn(invented_scene, result[0])

    def test_strict_mode_rejects_three_invalid_replies_without_local_invention(self):
        with mock.patch.object(refiner, '_request_ollama', side_effect=['invalid'] * 3) as request:
            with self.assertRaisesRegex(RuntimeError, 'strict mode rejected all three responses'):
                refiner.OnekoOllamaPromptRefiner().refine(**refine_kwargs(fallback_mode='strict'))
        self.assertEqual(request.call_count, 3)

    def test_continue_mode_preserves_raw_candidate_without_inventing_missing_label(self):
        raw = 'A red fox watches three blue bottles beside a quiet forest pool.'
        with mock.patch.object(refiner, '_request_ollama', side_effect=[raw] * 3) as request:
            result = refiner.OnekoOllamaPromptRefiner().refine(**refine_kwargs(fallback_mode='continue'))
        self.assertEqual(request.call_count, 3)
        self.assertIn(raw, result[0])
        self.assertNotIn(LABEL, result[0])
        self.assertIn('fallback_mode=continue', result[2])

    def test_continue_mode_keeps_connected_inputs_after_transport_error(self):
        with mock.patch.object(refiner, '_request_ollama', side_effect=RuntimeError('offline')) as request:
            result = refiner.OnekoOllamaPromptRefiner().refine(**refine_kwargs(fallback_mode='continue', left='two white birds'))
        self.assertEqual(request.call_count, 1)
        self.assertIn(SOURCE, result[0])
        self.assertIn('two white birds', result[0])
        self.assertIn('offline', result[2])

    def test_spatial_only_fallback_preserves_literal_case_punctuation_and_newlines(self):
        region = 'three brass signs reading "North_3,  OPEN\nCafé_7"'
        with mock.patch.object(refiner, '_request_ollama', side_effect=['invalid'] * 3):
            result = refiner.OnekoOllamaPromptRefiner().refine(**refine_kwargs(word_salad='', style_anchor='', left=region))
        self.assertIn('"North_3,  OPEN\nCafé_7"', result[0])
        self.assertIn('three brass signs', result[0])
        self.assertIn('left', result[0].lower())

    def test_region_visible_text_is_repaired_before_missing_region_append(self):
        region = 'three brass signs reading "NORTH_3"'
        bad = values(background_prompt='On the left, three brass signs read "SOUTH_3". Two birds watch the fox.')
        good = values(background_prompt='On the left, three brass signs read "NORTH_3". Two birds watch the fox.')
        with mock.patch.object(refiner, '_request_ollama', side_effect=[json.dumps(bad, ensure_ascii=False), json.dumps(good, ensure_ascii=False)]) as request:
            result = refiner.OnekoOllamaPromptRefiner().refine(**refine_kwargs(left=region))
        self.assertEqual(request.call_count, 2)
        self.assertIn('"NORTH_3"', result[0])
        self.assertNotIn('"SOUTH_3"', result[0])

    def test_plan_compile_accepts_empty_negative_and_preserves_required_elements(self):
        plan = {
            'subject': 'red fox', 'action': 'stands', 'environment': 'wooden arch',
            'style': ANCHOR, 'composition': 'fox beside three blue glass bottles',
            'lighting': 'gentle morning light', 'camera': 'eye level',
            'subject_details': ['three blue glass bottles'],
            'spatial_relations': ['Two white birds perch to the left of a wooden arch behind the fox'],
            'required_elements': ['red fox', 'three blue glass bottles', 'Two white birds', LABEL],
            'avoid': [], 'discarded_terms': [],
        }
        with mock.patch.object(refiner, '_request_ollama', side_effect=[json.dumps(plan, ensure_ascii=False), json.dumps(values(), ensure_ascii=False)]) as request:
            result = refiner.OnekoOllamaPromptRefiner().refine(**refine_kwargs(pipeline_mode='plan_compile', fallback_mode='strict'))
        self.assertEqual(request.call_count, 2)
        self.assertEqual(result[1], '')
        self.assertIn(LABEL, result[0])
        self.assertIn('MOSS', request.call_args_list[1].kwargs['system_instructions'])
        self.assertEqual(json.loads(result[6])['subject'], 'red fox')
        self.assertIn('final_local_issues=0', result[2])

    def test_all_single_attempts_receive_moss_system_rules_without_forced_text_negatives(self):
        replies = ['invalid', 'still invalid', json.dumps(values(), ensure_ascii=False)]
        with mock.patch.object(refiner, '_request_ollama', side_effect=replies) as request:
            result = refiner.OnekoOllamaPromptRefiner().refine(**refine_kwargs())
        self.assertEqual(request.call_count, 3)
        self.assertEqual(result[1], '')
        for call in request.call_args_list:
            system = call.kwargs['system_instructions']
            self.assertIn('MOSS', system)
            self.assertRegex(system.lower(), r'(?:negative[^\n]*(?:empty|optional)|(?:empty|optional)[^\n]*negative)')
            self.assertIn('original language', system.lower())
            self.assertNotIn('text/watermark/logo', system)
        legacy = refiner._prompt_system_instructions('strict')
        self.assertEqual(legacy, refiner.SYSTEM_INSTRUCTIONS + '\n\n' + refiner._prompt_mode_instructions('strict'))

    def test_quoted_style_with_text_carrier_is_not_promoted_to_image_lettering(self):
        source = 'A poster in the style of "watercolor", no lettering.'
        response = values(
            base_prompt='A soft watercolor uses gentle morning light.',
            foreground_prompt='A red fox stands beside three blue bottles on a decorative poster.',
            background_prompt='Leaves and pale clouds frame the fox.',
        )
        with mock.patch.object(refiner, '_request_ollama', return_value=json.dumps(response)) as request:
            result = refiner.OnekoOllamaPromptRefiner().refine(**refine_kwargs(word_salad=source, style_anchor=''))
        self.assertEqual(request.call_count, 1)
        self.assertIn('watercolor', result[0])
        self.assertNotIn('"watercolor"', result[0])

    def test_avoided_visible_text_is_not_forced_into_positive_output(self):
        source = 'A red fox beside a blank brass sign. Avoid text reading "STOP".'
        response = values(
            foreground_prompt='A red fox stands beside a blank brass sign and three blue bottles.',
            background_prompt='Two white birds perch on the arch behind the fox.',
            negative='unwanted lettering, "STOP"',
        )
        with mock.patch.object(refiner, '_request_ollama', return_value=json.dumps(response)) as request:
            result = refiner.OnekoOllamaPromptRefiner().refine(**refine_kwargs(word_salad=source))
        self.assertEqual(request.call_count, 1)
        self.assertNotIn('"STOP"', result[0])
        self.assertIn('"STOP"', result[1])

    def test_punctuation_in_first_quote_does_not_hide_second_required_label(self):
        source = 'Two signs read "Hello!" and "Goodbye!".'
        bad = values(foreground_prompt='Two wooden signs read "Hello!" and "Farewell!".')
        with self.assertRaisesRegex(ValueError, 'visible text'):
            self.postprocess(bad, source=source)
        good = values(foreground_prompt='Two wooden signs read "Hello!" and "Goodbye!".')
        result = self.postprocess(good, source=source)
        self.assertIn('"Hello!"', result[0])
        self.assertIn('"Goodbye!"', result[0])

    def test_exception_to_no_text_still_requires_its_explicit_label(self):
        source = 'No text except a sign reading "WELCOME".'
        bad = values(foreground_prompt='A red fox stands beside a blank wooden sign.')
        with self.assertRaisesRegex(ValueError, 'visible text'):
            self.postprocess(bad, source=source)
        good = values(foreground_prompt='A red fox stands beside a wooden sign reading "WELCOME".')
        result = self.postprocess(good, source=source)
        self.assertIn('"WELCOME"', result[0])

    def test_cache_distinguishes_profile_and_all_fallback_modes(self):
        hashes = {refiner.OnekoOllamaPromptRefiner.IS_CHANGED(**refine_kwargs(fallback_mode=mode))
                  for mode in ('adaptive', 'strict', 'continue')}
        self.assertEqual(len(hashes), 3)
        old = refiner.OnekoOllamaPromptRefiner.IS_CHANGED(**refine_kwargs(target_profile='krea2'))
        self.assertNotIn(old, hashes)


class MossPromptLanguageTests(unittest.TestCase):
    def test_german_normalization_then_generation_preserves_visible_text_and_fixed_anchor(self):
        source = 'Ein roter Fuchs neben drei blauen Glasflaschen. Das Schild zeigt "Grüße: Café_7 — 20%".'
        english = 'A red fox beside three blue glass bottles. The sign reads "Grüße: Café_7 — 20%".'
        translated = refiner._language_source_values(english)
        with mock.patch.object(refiner, '_request_ollama', side_effect=[json.dumps(translated, ensure_ascii=False), json.dumps(values(), ensure_ascii=False)]) as request:
            result = refiner.OnekoOllamaPromptRefiner().refine(**refine_kwargs(word_salad=source))
        self.assertEqual(request.call_count, 2)
        self.assertIn(english, request.call_args_list[1].args[2])
        self.assertNotIn('Ein roter Fuchs', request.call_args_list[1].args[2])
        self.assertIn(LABEL, result[0])
        self.assertTrue(result[3].startswith(ANCHOR))
        self.assertIn('returned revised source fields for English normalization', result[2])

    def test_language_repair_prevents_translating_the_required_lettering(self):
        source = 'Ein roter Fuchs. Das Schild zeigt "Grüße: Café_7 — 20%".'
        invalid = refiner._language_source_values('A red fox. The sign reads "Greetings: Cafe 7".')
        valid = refiner._language_source_values('A red fox. The sign reads "Grüße: Café_7 — 20%".')
        replies = [json.dumps(invalid), json.dumps(valid, ensure_ascii=False), json.dumps(values(), ensure_ascii=False)]
        with mock.patch.object(refiner, '_request_ollama', side_effect=replies) as request:
            result = refiner.OnekoOllamaPromptRefiner().refine(**refine_kwargs(word_salad=source))
        self.assertEqual(request.call_count, 3)
        self.assertIn(LABEL, result[0])
        self.assertNotIn('"Greetings: Cafe 7"', result[0])
        self.assertIn('language-normalization', request.call_args_list[1].args[2])

    def test_changed_multiline_curly_lettering_after_translation_obeys_fallback_mode(self):
        label = '„Grüße:\nCafé_7“'
        source = f'Ein roter Fuchs hält ein Schild mit der Aufschrift {label}.'
        invalid_translation = refiner._language_source_values(
            'A red fox holds a sign reading „Greetings:\nCafe 7“.'
        )
        good_response = values(foreground_prompt=f'A red fox holds a brass sign reading {label}.')
        for mode in ('strict', 'adaptive', 'continue'):
            with self.subTest(mode=mode):
                replies = [json.dumps(invalid_translation, ensure_ascii=False)]
                if mode != 'strict':
                    replies.append(json.dumps(good_response, ensure_ascii=False))
                with mock.patch.object(refiner, '_request_ollama', side_effect=replies) as request:
                    if mode == 'strict':
                        with self.assertRaisesRegex(RuntimeError, 'visible text'):
                            refiner.OnekoOllamaPromptRefiner().refine(**refine_kwargs(word_salad=source, fallback_mode=mode))
                        self.assertEqual(request.call_count, 1)
                    else:
                        result = refiner.OnekoOllamaPromptRefiner().refine(**refine_kwargs(word_salad=source, fallback_mode=mode))
                        self.assertEqual(request.call_count, 2)
                        self.assertIn(source, request.call_args_list[1].args[2])
                        self.assertIn(label, result[0])
                        self.assertNotIn('Greetings:', result[0])
                        self.assertIn('original source fields were passed through unchanged', result[2])

    def test_unchanged_german_language_reply_is_not_reported_as_verified_english(self):
        source = 'Ein roter Fuchs neben einem Schild mit der Aufschrift "Grüße: Café_7 — 20%".'
        unchanged = refiner._language_source_values(source)
        replies = [json.dumps(unchanged, ensure_ascii=False), json.dumps(values(), ensure_ascii=False)]
        with mock.patch.object(refiner, '_request_ollama', side_effect=replies) as request:
            result = refiner.OnekoOllamaPromptRefiner().refine(**refine_kwargs(word_salad=source))
        self.assertEqual(request.call_count, 2)
        self.assertIn('source language was not independently verified', result[2])
        self.assertNotIn('detected english', result[2])
        self.assertNotIn('translated source text to English', result[2])
        self.assertIn(LABEL, result[0])

    def test_revised_moss_report_does_not_claim_verification_or_change_legacy_report(self):
        result = ('A red fox.', '', 'Kept the fox.', '', 'A red fox.', '')
        moss = refiner._with_language_report(result, 'german', True, target_profile='moss_image')
        legacy = refiner._with_language_report(result, 'german', True)
        self.assertIn('returned revised source fields for English normalization', moss[2])
        self.assertNotIn('detected german', moss[2])
        self.assertNotIn('translated source text to English', moss[2])
        self.assertIn('detected german and translated source text to English', legacy[2])
        self.assertEqual(moss[:2] + moss[3:], result[:2] + result[3:])


if __name__ == '__main__':
    unittest.main()
