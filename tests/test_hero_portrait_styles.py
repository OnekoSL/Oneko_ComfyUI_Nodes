"""Regression coverage for distinct, locally enforced portrait render styles."""

import json
from unittest.mock import patch

import pytest

from Oneko_ComfyUI_Nodes.nodes import hero_portrait_compose as compose
from Oneko_ComfyUI_Nodes.nodes import ollama_hero_portrait as portrait


STYLE_OPTIONS = (
    "Fantasy-Illustration", "Realistische Malerei", "Fotorealistisch", "Ölgemälde",
    "Aquarell", "Anime", "Comic", "Eigene Vorgabe",
)
STYLE_MARKERS = {
    "Fantasy-Illustration": "Painterly fantasy book illustration",
    "Realistische Malerei": "Realistic painted portrait",
    "Fotorealistisch": "Photorealistic portrait",
    "Ölgemälde": "Traditional oil painting",
    "Aquarell": "Watercolor",
    "Anime": "Anime character illustration",
    "Comic": "Comic illustration",
}


def compiled(style, language="English", prompt_format="Natürliche Sprache", details=""):
    hero = {
        "source_format": "helden_xml", "species": "Amaunir", "gender": "weiblich",
        "appearance": {"age": "20", "hair_color": "schwarz", "eye_color": "gelb"},
    }
    task = {
        "output_language": language, "art_style": {"selection": style},
        "prompt_format": {"selection": prompt_format}, "additional_details": details,
        "pose": "Steht neben einem Baum.", "background": "Ein Marktplatz.",
    }
    plan = compose.build_portrait_plan(hero, task)
    localized = dict(plan["fields"])
    localized["identity"] = "CHARACTER_SOURCE_20"
    localized["pose"] = "POSE_SOURCE"
    localized["background"] = "BACKGROUND_SOURCE"
    return compose.compile_portrait(plan, localized, [], "", task)


def test_saved_style_options_and_default_are_unchanged():
    with patch.object(portrait, "_available_ollama_models", return_value=["local-model"]):
        schema = portrait.OnekoOllamaHeroPortrait.define_schema()
    style = next(item for item in schema.inputs if item.id == "art_style")
    assert tuple(style.options) == STYLE_OPTIONS
    assert style.default == "Fantasy-Illustration"
    assert tuple(portrait.ART_STYLES) == STYLE_OPTIONS


@pytest.mark.parametrize("style", STYLE_MARKERS)
@pytest.mark.parametrize("language", ["English", "German"])
@pytest.mark.parametrize("prompt_format", ["Natürliche Sprache", "Tags"])
def test_preset_medium_precedes_character_and_preserves_portrait_facts(style, language, prompt_format):
    positive, _, _ = compiled(style, language, prompt_format)
    fixed_style = compose._ART_STYLE[style][language == "German"]
    assert fixed_style
    assert positive.startswith(fixed_style)
    assert positive.count(fixed_style) == 1
    assert positive.index(fixed_style) < positive.index("CHARACTER_SOURCE_20")
    assert "POSE_SOURCE" in positive and "BACKGROUND_SOURCE" in positive
    if language == "English":
        assert positive.startswith(STYLE_MARKERS[style])
        assert "black hair" in positive and "yellow eyes" in positive
        assert "catlike face and fur" in positive
    else:
        assert fixed_style != compose._ART_STYLE[style][0]
        assert "Haarfarbe: schwarz" in positive and "Augenfarbe: gelb" in positive
        assert "katzenartigem Gesicht und Fell" in positive


def test_fantasy_is_painterly_and_excludes_the_anime_look_without_humanizing_species():
    positive, negative, _ = compiled("Fantasy-Illustration")
    assert "brush texture" in positive
    assert "species and age" in positive
    assert "catlike face and fur" in positive
    assert all(term in negative.casefold() for term in ("anime", "manga", "cel shading"))
    assert not any(term in positive.casefold() for term in ("human face", "human skin", "anime"))


@pytest.mark.parametrize("style", ["Anime", "Comic", "Eigene Vorgabe"])
@pytest.mark.parametrize("language", ["English", "German"])
def test_stylized_and_custom_choices_are_not_negated(style, language):
    positive, negative, _ = compiled(style, language, details="Manga illustration")
    assert "Manga illustration" in positive
    assert not any(term in negative.casefold() for term in ("anime", "manga", "cel shading"))


@pytest.mark.parametrize("language", ["English", "German"])
@pytest.mark.parametrize("prompt_format", ["Natürliche Sprache", "Tags"])
def test_custom_style_retains_user_medium_without_a_conflicting_preset(language, prompt_format):
    details = "Pixel art in a limited four-color palette"
    positive, negative, _ = compiled("Eigene Vorgabe", language, prompt_format, details)
    assert details in positive
    assert not any(marker in positive for marker in STYLE_MARKERS.values())
    expected = "mehrfach dargestellte Figur, Collage, Wasserzeichen" if language == "German" else "duplicate character, collage, watermark"
    assert negative == expected


@pytest.mark.parametrize("language,expected", [
    ("English", "Character illustration"), ("German", "Charakterillustration"),
])
def test_empty_custom_style_keeps_neutral_fallback(language, expected):
    positive, _, _ = compiled("Eigene Vorgabe", language)
    assert expected in positive
    assert "Painterly fantasy" not in positive and "Anime character" not in positive


@pytest.mark.parametrize("style", STYLE_MARKERS)
def test_ollama_receives_the_same_style_that_reaches_the_image_prompt(style):
    captured = []

    def response(*args, **kwargs):
        payload = json.loads(args[2])
        captured.append(payload)
        return json.dumps({
            "translations": payload["source_fields"], "equipment_ids": [],
            "omitted_fact_ids": [], "embellishment": "", "report": "",
        })

    with patch.object(portrait, "_request_ollama", side_effect=response), \
         patch.object(portrait, "_unload_after_run"):
        positive, _, _, _ = portrait.OnekoOllamaHeroPortrait.execute(prompt_detail="Kompakt",
            document_text="A 20-year-old elf wearing a green cloak.", art_style=style,
        ).result[:4]
    assert len(captured) == 1
    requested = captured[0]["task"]["art_style"]
    assert requested["selection"] == style
    assert requested["instruction"] == compose._ART_STYLE[style][0]
    assert positive.startswith(requested["instruction"])
