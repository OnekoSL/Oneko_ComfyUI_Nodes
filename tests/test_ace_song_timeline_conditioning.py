import json
import math
import sys
import unittest
from pathlib import Path

from tests.paths import COMFY_ROOT, ONEKO_ROOT
CUSTOM_NODE_ROOT = ONEKO_ROOT
for import_path in (COMFY_ROOT, CUSTOM_NODE_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from Oneko_ComfyUI_Nodes.nodes import ace_song_timeline_conditioning as timeline
import torch
from comfy import conds, samplers


class FakeClip:
    def __init__(self):
        self.tokenize_calls = []
        self.encode_calls = []

    def tokenize(self, tags, **kwargs):
        value = {"tags": tags, **kwargs}
        self.tokenize_calls.append(value)
        return value

    def encode_from_tokens_scheduled(self, tokens):
        self.encode_calls.append(tokens)
        index = len(self.encode_calls)
        return [
            [
                torch.full((1, 4, 8), float(index)),
                {"conditioning_lyrics": torch.full((1, 3, 8), float(index))},
            ]
        ]


def _base_conditioning(code_count=150, strength=0.8):
    return [
        [
            torch.ones((1, 4, 8)),
            {"audio_codes": [list(range(code_count))], "strength": strength},
        ]
    ]


def _plan():
    return json.dumps(
        {
            "status": "ok",
            "arranged_sections": [
                {
                    "id": "S01",
                    "origin": "source",
                    "header": "Intro",
                    "direction": "quiet lute establishes the pulse",
                    "lyrics": "",
                },
                {
                    "id": "S02",
                    "origin": "source",
                    "header": "Verse 1",
                    "direction": "restrained tenor over sparse lute",
                    "lyrics": "In Notmark zieht der kalte Wind.",
                },
                {
                    "id": "A01",
                    "origin": "additional",
                    "header": "Final Chorus",
                    "direction": "loud communal voices and driving drum",
                    "lyrics": "Die Warzensau holt der Wind geschwind.",
                },
            ],
        },
        ensure_ascii=False,
    )


class AceSongTimelineConditioningTests(unittest.TestCase):
    def test_registration_and_public_contract(self):
        self.assertIs(
            timeline.NODE_CLASS_MAPPINGS["OnekoAceSongTimelineConditioning"],
            timeline.OnekoAceSongTimelineConditioning,
        )
        self.assertEqual(
            timeline.NODE_DISPLAY_NAME_MAPPINGS["OnekoAceSongTimelineConditioning"],
            "ACE Song Timeline Conditioning (Oneko)",
        )
        node = timeline.OnekoAceSongTimelineConditioning
        self.assertEqual(node.CATEGORY, "Oneko/08 Audio/ACE")
        self.assertEqual(
            node.RETURN_NAMES,
            ("conditioning", "timeline_json", "report", "duration_seconds"),
        )
        inputs = node.INPUT_TYPES()["required"]
        self.assertEqual(inputs["base_strength"][1]["default"], 0.35)
        self.assertEqual(inputs["region_strength"][1]["default"], 1.0)
        self.assertEqual(inputs["transition_seconds"][1]["default"], 1.0)
        self.assertEqual(inputs["minimum_section_seconds"][1]["default"], 4.0)

    def test_section_source_priority_and_header_direction(self):
        sections, source, warnings = timeline._resolve_sections("[Verse]\nIgnored", _plan())
        self.assertEqual(source, "director_plan")
        self.assertFalse(warnings)
        self.assertEqual([item["id"] for item in sections], ["S01", "S02", "A01"])

        lyrics = "[Verse 1 | VARIATION: quiet lute and restrained tenor]\nEine Zeile.\n\n[Chorus]\nEin Ruf."
        sections, source, warnings = timeline._resolve_sections(lyrics, "not json")
        self.assertEqual(source, "structured_lyrics")
        self.assertTrue(warnings)
        self.assertEqual(sections[0]["header"], "Verse 1")
        self.assertEqual(sections[0]["direction"], "quiet lute and restrained tenor")

    def test_stanza_detection_is_available_without_director(self):
        lyrics = """Eine Strophe im Wind.
Noch eine Zeile klingt.

Hebt an und singt es laut.
Der Baron hat uns beraubt.

Eine zweite Strophe klingt.
Bis sich die Nacht verschlingt.

Hebt an und singt es laut.
Der Baron hat uns beraubt."""
        sections, source, _ = timeline._resolve_sections(lyrics, "")
        self.assertEqual(source, "stanza_detection")
        self.assertEqual([item["header"] for item in sections], ["Verse 1", "Chorus", "Verse 2", "Final Chorus"])

    def test_audio_code_validation(self):
        codes = timeline._extract_audio_codes(_base_conditioning(20))
        self.assertEqual(len(codes[0]), 20)
        with self.assertRaisesRegex(ValueError, "requires generated"):
            timeline._extract_audio_codes([[torch.ones(1), {}]])
        with self.assertRaisesRegex(ValueError, "non-rectangular"):
            timeline._extract_audio_codes([[torch.ones(1), {"audio_codes": [[1, 2], [3]]}]])
        with self.assertRaisesRegex(ValueError, "already regionalized"):
            timeline._extract_audio_codes([[torch.ones(1), {"audio_codes": [[1]], "area": (1, 0)}]])

    def test_integer_allocation_and_duration_overrides(self):
        sections = timeline._sections_from_plan(_plan())
        allocations, warnings = timeline._allocate_codes(sections, 150, 4.0, "S01=6.1")
        self.assertEqual(sum(allocations), 150)
        self.assertEqual(allocations[0], 31)
        self.assertTrue(warnings)
        with self.assertRaisesRegex(ValueError, "unknown section"):
            timeline._allocate_codes(sections, 150, 4.0, "S99=4")
        with self.assertRaisesRegex(ValueError, "exceed"):
            timeline._allocate_codes(sections, 30, 4.0, "S01=8")

    def test_builds_regions_without_generating_new_audio_codes(self):
        clip = FakeClip()
        base = _base_conditioning()
        original_tensor = base[0][0].clone()
        original_metadata = base[0][1].copy()
        result, timeline_json, report, duration = timeline.OnekoAceSongTimelineConditioning().build_timeline(
            clip=clip,
            base_conditioning=base,
            tags="dark Irish pub folk",
            lyrics="unused",
            plan_json=_plan(),
            base_strength=0.35,
            region_strength=1.2,
            transition_seconds=1.0,
            minimum_section_seconds=4.0,
            duration_overrides="S01=6",
        )

        self.assertEqual(duration, 30.0)
        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(result[0][1]["strength"], 0.28)
        self.assertTrue(torch.equal(base[0][0], original_tensor))
        self.assertEqual(base[0][1], original_metadata)
        self.assertEqual(len(clip.tokenize_calls), 3)
        self.assertTrue(all(call["generate_audio_codes"] is False for call in clip.tokenize_calls))
        self.assertTrue(all("Current time region" in call["tags"] for call in clip.tokenize_calls))
        self.assertTrue(all(item[1]["mask"].shape == (1, 750) for item in result[1:]))
        self.assertTrue(all(len(item[1]["area"]) == 2 for item in result[1:]))
        self.assertTrue(all(item[1]["strength"] == 1.2 for item in result[1:]))
        for item in result[1:]:
            area_length, area_start = item[1]["area"]
            code_start = area_start // timeline.LATENTS_PER_AUDIO_CODE
            code_end = math.ceil((area_start + area_length) / timeline.LATENTS_PER_AUDIO_CODE)
            self.assertEqual(item[1]["audio_codes"][0], list(range(code_start, code_end)))
        payload = json.loads(timeline_json)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["source"], "director_plan")
        self.assertEqual(payload["sections"][-1]["end"], 30.0)
        self.assertIn("3 regions", report)

        first = result[1][1]
        second = result[2][1]
        overlap_start = max(first["area"][1], second["area"][1])
        overlap_end = min(
            first["area"][1] + first["area"][0],
            second["area"][1] + second["area"][0],
        )
        summed = first["mask"][:, overlap_start:overlap_end] + second["mask"][:, overlap_start:overlap_end]
        self.assertTrue(torch.allclose(summed, torch.ones_like(summed), atol=1e-6))

    def test_single_section_uses_base_only_at_full_strength(self):
        clip = FakeClip()
        result, timeline_json, report, duration = timeline.OnekoAceSongTimelineConditioning().build_timeline(
            clip=clip,
            base_conditioning=_base_conditioning(50, strength=0.8),
            tags="solo lute",
            lyrics="Nur ein zusammenhängender Liedtext.",
            base_strength=0.1,
            region_strength=2.0,
            transition_seconds=1.0,
            minimum_section_seconds=4.0,
            duration_overrides="",
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1]["strength"], 0.8)
        self.assertFalse(clip.tokenize_calls)
        self.assertEqual(json.loads(timeline_json)["status"], "base_only")
        self.assertEqual(duration, 10.0)
        self.assertIn("skipped", report)

    def test_rejects_more_than_maximum_sections(self):
        plan = {
            "status": "ok",
            "arranged_sections": [
                {
                    "id": f"S{index + 1:02d}",
                    "origin": "source",
                    "header": f"Verse {index + 1}",
                    "direction": "steady lute accompaniment",
                    "lyrics": f"Zeile {index + 1}",
                }
                for index in range(timeline.MAX_SECTIONS + 1)
            ],
        }
        with self.assertRaisesRegex(ValueError, "at most"):
            timeline.OnekoAceSongTimelineConditioning().build_timeline(
                clip=FakeClip(),
                base_conditioning=_base_conditioning(1000),
                tags="folk",
                lyrics="unused",
                plan_json=json.dumps(plan),
                base_strength=0.35,
                region_strength=1.0,
                transition_seconds=1.0,
                minimum_section_seconds=0.2,
                duration_overrides="",
            )

    def test_sampler_accepts_generated_one_dimensional_area_and_mask(self):
        clip = FakeClip()
        result, _, _, _ = timeline.OnekoAceSongTimelineConditioning().build_timeline(
            clip=clip,
            base_conditioning=_base_conditioning(100),
            tags="folk",
            lyrics="[Verse 1]\nEins.\n\n[Chorus]\nZwei.",
            base_strength=0.35,
            region_strength=1.0,
            transition_seconds=1.0,
            minimum_section_seconds=4.0,
            duration_overrides="",
        )
        metadata = result[1][1].copy()
        metadata["uuid"] = "timeline-test"
        metadata["model_conds"] = {
            "c_crossattn": conds.CONDRegular(result[1][0]),
            "audio_codes": conds.CONDRegular(torch.tensor(metadata["audio_codes"])),
        }
        latent = torch.zeros((1, 64, 500))
        samplers.resolve_areas_and_cond_masks_multidim([metadata], latent.shape[2:], torch.device("cpu"))
        processed = samplers.get_area_and_mult(metadata, latent, torch.ones(1))
        self.assertEqual(processed.input_x.shape[-1], metadata["area"][0])
        self.assertEqual(processed.mult.shape, processed.input_x.shape)


if __name__ == "__main__":
    unittest.main()
