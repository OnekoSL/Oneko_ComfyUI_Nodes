"""Behavior checks for expanded portrait prose and source-bound term explanations."""

import asyncio
import json
from unittest.mock import patch

import pytest
import execution
import nodes as core_nodes
from comfy_extras.nodes_preview_any import PreviewAny

from Oneko_ComfyUI_Nodes.nodes import ollama_hero_portrait as portrait
from Oneko_ComfyUI_Nodes.nodes.hero_document import load_hero_document
from Oneko_ComfyUI_Nodes.nodes.hero_portrait_compose import build_portrait_plan, compile_portrait


HERO_XML = """<helden><held name="Alinde Ehrwald"><basis>
<geschlecht name="weiblich"/>
<rasse string="Mittelländerin"><aussehen alter="32" augenfarbe="dunkelbraun"
 haarfarbe="kirschrot"/></rasse>
<ausbildungen><ausbildung string="Nebelweberin"/></ausbildungen>
</basis><gegenstände><gegenstand name="Armreif" anzahl="1"/></gegenstände>
</held></helden>"""


@pytest.fixture
def transport():
    overrides = {}

    def response(*args, **kwargs):
        prompt = args[2]
        if "Original task:\n" in prompt:
            prompt = prompt.split("Original task:\n", 1)[1].split("\n\nInvalid response", 1)[0]
        payload = json.loads(prompt)
        if "descriptions" in payload:
            if request.expansion_error is not None:
                raise request.expansion_error
            if request.expansion_response is not None:
                return request.expansion_response
            return json.dumps({
                "descriptions": {**payload["descriptions"], **request.expansion_overrides},
            }, ensure_ascii=False)
        localized = {**payload["source_fields"], **payload.get("canonical_translations", {}), **overrides}
        return json.dumps({
            "translations": localized,
            "equipment_ids": [key for key in localized if key.startswith("equipment_")],
            "omitted_fact_ids": request.omitted_fact_ids, "embellishment": "", "report": "",
        }, ensure_ascii=False)

    with patch.object(portrait, "_request_ollama", side_effect=response) as request, \
         patch.object(portrait, "_unload_after_run"):
        request.expansion_overrides = {}
        request.expansion_error = None
        request.expansion_response = None
        request.omitted_fact_ids = []
        yield request, overrides


def schema():
    with patch.object(portrait, "_available_ollama_models", return_value=["local-model"]):
        return portrait.OnekoOllamaHeroPortrait.define_schema()


def test_detail_and_glossary_widgets_follow_all_existing_optional_inputs():
    definition = schema()
    required = [item.id for item in definition.inputs if not item.optional]
    optional = [item.id for item in definition.inputs if item.optional]
    assert required == [
        "pose", "background", "additional_details", "framing", "view", "art_style",
        "expression", "lighting", "equipment_mode", "detail_policy", "prompt_format",
        "output_language", "ollama_url", "ollama_model", "seed", "temperature", "top_p",
        "context_length", "timeout_seconds", "unload_after_run",
    ]
    assert optional == ["document_path", "document_text", "hero_name", "prompt_detail", "term_explanations"]
    inputs = {item.id: item for item in definition.inputs}
    assert inputs["prompt_detail"].options == ["Kompakt", "Ausführlich", "Sehr ausführlich"]
    assert inputs["prompt_detail"].default == "Ausführlich"
    assert inputs["term_explanations"].default == ""
    assert inputs["term_explanations"].multiline
    assert inputs["term_explanations"].dynamic_prompts is False
    assert [output.id for output in definition.outputs] == [
        "positive", "negative", "character_summary", "report", "filename_hero_name",
    ]


def test_native_validation_accepts_old_api_prompt_without_new_optional_fields():
    inputs = {item.id: item.default for item in schema().inputs
              if item.id not in ("prompt_detail", "term_explanations")}
    inputs["document_text"] = HERO_XML
    graph = {
        "1": {"class_type": "OnekoOllamaHeroPortrait", "inputs": inputs},
        "2": {"class_type": "PreviewAny", "inputs": {"source": ["1", 0]}},
    }
    with patch.object(portrait, "_available_ollama_models", return_value=["local-model"]), \
         patch.dict(core_nodes.NODE_CLASS_MAPPINGS, {
             "OnekoOllamaHeroPortrait": portrait.OnekoOllamaHeroPortrait, "PreviewAny": PreviewAny,
         }):
        valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt("portrait-detail-legacy", graph, None))
    assert valid, (error, node_errors)
    assert outputs == ["2"]
    assert node_errors == {}


@pytest.mark.parametrize("detail", ["Kompakt", "Ausführlich", "Sehr ausführlich"])
@pytest.mark.parametrize("prompt_format", ["Natürliche Sprache", "Tags"])
def test_selected_detail_reaches_ollama_and_local_report_without_changing_response_contract(detail, prompt_format, transport):
    request, _ = transport
    result = portrait.OnekoOllamaHeroPortrait.execute(
        document_text=HERO_XML, prompt_detail=detail, prompt_format=prompt_format,
        pose="Steht aufrecht.", background="Ein Marktplatz im Tageslicht.",
    ).result
    assert request.call_count == (1 if detail == "Kompakt" else 2)
    first = request.call_args_list[0]
    payload = json.loads(first.args[2])
    assert payload["task"]["prompt_detail"]["selection"] == detail
    assert payload["task"]["prompt_detail"]["instruction"].strip()
    assert payload["task"]["prompt_format"]["selection"] == prompt_format
    assert set(first.kwargs["output_schema"]["required"]) == {
        "translations", "equipment_ids", "omitted_fact_ids", "embellishment", "report",
    }
    if detail != "Kompakt":
        expanded = json.loads(request.call_args.args[2])
        assert expanded["detail"]["selection"] == detail
        assert expanded["format"] == prompt_format
        assert set(expanded["descriptions"]) == {"pose", "background"}
        assert request.call_args.kwargs["output_schema"]["required"] == ["descriptions"]
        assert "Ollama-Ausarbeitung für 2 Textfelder übernommen." in result[3]
    assert detail in result[3]
    assert len(result) == 5
    assert result[4] == "Alinde_Ehrwald"


def test_missing_detail_argument_uses_expanded_default(transport):
    request, _ = transport
    portrait.OnekoOllamaHeroPortrait.execute(document_text=HERO_XML, pose="Steht aufrecht.")
    assert request.call_count == 2
    payload = json.loads(request.call_args_list[0].args[2])
    assert payload["task"]["prompt_detail"]["selection"] == "Ausführlich"


def test_unknown_detail_is_rejected_before_ollama(transport):
    request, _ = transport
    with pytest.raises(ValueError, match="prompt_detail"):
        portrait.OnekoOllamaHeroPortrait.execute(document_text=HERO_XML, prompt_detail="Beliebig")
    request.assert_not_called()


@pytest.mark.parametrize("output_language", ["Deutsch", "Englisch"])
def test_multisentence_pose_and_background_survive_without_clipping(output_language, transport):
    request, _ = transport
    pose = (
        "The character stands upright with the upper body turned slightly to the side. "
        "The shoulders follow that gentle turn without hiding the face. "
        "The attentive gaze remains directed toward the viewer. "
        "The posture is relaxed and steady, with the same quiet stance maintained throughout the composition."
    )
    background = (
        "A bustling market occupies the space behind the character. "
        "Its activity stays softly out of focus so the person remains the central subject. "
        "Warm daylight illuminates the scene consistently from the market into the portrait. "
        "The transition into the blurred background remains gradual, preserving the impression of a busy place "
        "while allowing the silhouette to remain distinct and readable at the front of the image."
    )
    request.expansion_overrides.update(pose=pose, background=background)
    result = portrait.OnekoOllamaHeroPortrait.execute(
        document_text=HERO_XML, pose="Entspannt aufrecht, leicht gedreht, Blick zur Kamera.",
        background="Belebter Marktplatz, weich unscharf, warmes Tageslicht.",
        output_language=output_language, prompt_detail="Sehr ausführlich",
    ).result
    assert request.call_count == 2
    assert pose in result[0]
    assert background in result[0]
    assert "Pose übernommen" in result[3]
    assert "Hintergrund übernommen" in result[3]


def test_expanded_mode_preserves_canonical_colors_against_incorrect_model_text(transport):
    request, overrides = transport
    overrides.update(hair="Blue hair", eyes="Violet eyes", equipment_0="a sword")
    positive = portrait.OnekoOllamaHeroPortrait.execute(
        document_text=HERO_XML, prompt_detail="Sehr ausführlich", pose="Steht aufrecht.",
    ).result[0]
    assert request.call_count == 2
    expansion = json.loads(request.call_args.args[2])
    assert set(expansion["descriptions"]) == {"pose"}
    assert "cherry" in expansion["reference_facts"]["hair"]
    assert expansion["reference_facts"]["eyes"] == "dark brown eyes"
    assert expansion["reference_facts"]["equipment_0"] == "bracelet"
    assert "cherry" in positive.casefold() and "hair" in positive.casefold()
    assert "dark brown eyes" in positive.casefold()
    assert "bracelet" in positive.casefold()
    assert "blue hair" not in positive.casefold()
    assert "violet eyes" not in positive.casefold()
    assert "sword" not in positive.casefold()


def test_custom_profession_definition_is_source_data_and_keeps_hero_summary_and_filename(transport):
    request, _ = transport
    definition = "Eine Weberin, die leichte Stoffe herstellt."
    explanations = "# Eigener Begriff\nNebelweberin = " + definition
    result = portrait.OnekoOllamaHeroPortrait.execute(
        document_text=HERO_XML, term_explanations=explanations, output_language="Deutsch",
    ).result
    payload = json.loads(request.call_args.args[2])
    assert payload["task"]["term_explanations"] == explanations
    assert payload["source_fields"]["profession_0"] == definition
    assert "Nebelweberin" not in payload["source_fields"]["identity"]
    assert definition.rstrip(".") in result[0]
    summary = json.loads(result[2])
    assert summary["professions"] == ["Nebelweberin"]
    assert summary == load_hero_document(document_text=HERO_XML)
    assert result[4] == "Alinde_Ehrwald"
    assert "Begriffe aufgelöst:" in result[3]
    assert "Nebelweberin" in result[3]


def test_custom_gloss_notes_are_reported_only_when_the_fact_is_used():
    hero = load_hero_document(document_text=HERO_XML)
    task = {
        "output_language": "German", "prompt_detail": {"selection": "Ausführlich"},
        "term_explanations": "Nebelweberin = Eine Weberin leichter Stoffe.",
    }
    plan = build_portrait_plan(hero, task)
    assert "profession_0" in plan["term_notes"]
    positive, _, report = compile_portrait(
        plan, plan["fields"], [], "", task, omitted_fact_ids=["profession_0"],
    )
    assert "Weberin leichter Stoffe" not in positive
    assert "Nebelweberin" not in report


def test_unknown_profession_specialization_is_visible_as_unresolved_in_the_report(transport):
    source = HERO_XML.replace('string="Nebelweberin"', 'string="Hexe: Nebelweberin"')
    result = portrait.OnekoOllamaHeroPortrait.execute(document_text=source).result
    assert "Ungeklärte Fachbegriffe:" in result[3]
    assert "Nebelweberin" in result[3]


def test_detail_and_term_explanations_change_cache_fingerprint():
    arguments = {"document_text": HERO_XML, "prompt_detail": "Kompakt", "term_explanations": ""}
    original = portrait.OnekoOllamaHeroPortrait.fingerprint_inputs(**arguments)
    assert original != portrait.OnekoOllamaHeroPortrait.fingerprint_inputs(
        **(arguments | {"prompt_detail": "Ausführlich"})
    )
    assert original != portrait.OnekoOllamaHeroPortrait.fingerprint_inputs(
        **(arguments | {"term_explanations": "Nebelweberin = Eine Weberin leichter Stoffe."})
    )


@pytest.mark.parametrize("detail", ["Ausführlich", "Sehr ausführlich"])
def test_expanded_modes_do_not_request_an_empty_second_pass(detail, transport):
    request, _ = transport
    result = portrait.OnekoOllamaHeroPortrait.execute(document_text=HERO_XML, prompt_detail=detail).result
    assert request.call_count == 1
    assert "Ollama-Ausarbeitung" not in result[3]
    assert "Ausarbeitung nicht übernommen" not in result[3]


def test_expansion_transport_failure_retains_first_pass_and_unloads_once(transport):
    request, overrides = transport
    overrides.update(pose="Standing upright beside the wall.")
    request.expansion_error = RuntimeError("Model connection unavailable during expansion")
    with patch.object(portrait, "_unload_after_run") as unload:
        result = portrait.OnekoOllamaHeroPortrait.execute(
            document_text=HERO_XML, pose="Steht aufrecht neben der Mauer.",
        ).result
    assert request.call_count == 2
    assert "Standing upright beside the wall." in result[0]
    assert "Ausarbeitung nicht übernommen; vollständige Erstbeschreibung erhalten:" in result[3]
    assert "Model connection unavailable during expansion" in result[3]
    assert "Reparaturversuch" not in result[3]
    unload.assert_called_once()


@pytest.mark.parametrize("expanded", [
    {},
    {"pose": "Standing on two steps."},
    {"pose": ""},
    {"pose": "Standing on 2 steps.", "identity": "A different person"},
])
def test_invalid_expansion_falls_back_to_complete_first_pass_without_retry(expanded, transport):
    request, overrides = transport
    overrides["pose"] = "Standing steadily on 2 steps."
    request.expansion_response = json.dumps({"descriptions": expanded})
    result = portrait.OnekoOllamaHeroPortrait.execute(
        document_text=HERO_XML, pose="Steht auf 2 Stufen.", prompt_detail="Sehr ausführlich",
    ).result
    assert request.call_count == 2
    assert "Standing steadily on 2 steps." in result[0]
    assert "A different person" not in result[0]
    assert "Ausarbeitung nicht übernommen; vollständige Erstbeschreibung erhalten:" in result[3]
    assert "Reparaturversuch" not in result[3]
    assert result[4] == "Alinde_Ehrwald"


def test_expansion_keeps_user_overrides_and_omitted_appearance_out_of_second_request(transport):
    request, _ = transport
    request.omitted_fact_ids = ["appearance_0"]
    source = HERO_XML.replace('haarfarbe="kirschrot"', 'haarfarbe="kirschrot" aussehentext0="Eine Narbe über dem Auge."')
    result = portrait.OnekoOllamaHeroPortrait.execute(
        document_text=source, pose="Steht aufrecht.", additional_details="Keine Narbe.",
    ).result
    assert request.call_count == 2
    expansion = json.loads(request.call_args.args[2])
    assert set(expansion["source_fields"]) == {"pose"}
    assert "appearance_0" not in expansion["reference_facts"]
    assert "additional_details" not in expansion["descriptions"]
    assert "Keine Narbe." in result[0]
    assert "Eine Narbe über dem Auge" not in result[0]


def test_plain_document_description_uses_the_focused_expansion_pass(transport):
    request, _ = transport
    request.expansion_overrides["description"] = (
        "A 30-year-old adult stands beside a tree. The person remains upright beside its trunk, "
        "with the body held in a relaxed standing position."
    )
    result = portrait.OnekoOllamaHeroPortrait.execute(
        document_text="A 30-year-old adult standing beside a tree.",
    ).result
    assert request.call_count == 2
    expansion = json.loads(request.call_args.args[2])
    assert set(expansion["descriptions"]) == {"description"}
    assert request.expansion_overrides["description"] in result[0]
    assert result[4] == "Unbenannter_Held"


@pytest.mark.parametrize("prompt_format,separator", [("Natürliche Sprache", ". "), ("Tags", ", ")])
def test_expansion_preserves_original_scene_anchor_when_place_and_light_are_omitted(prompt_format, separator, transport):
    request, overrides = transport
    anchor = "A bustling marketplace in Gareth under warm daylight"
    elaboration = "The market activity recedes into a softly blurred background behind the character"
    overrides["background"] = anchor
    request.expansion_overrides["background"] = elaboration
    result = portrait.OnekoOllamaHeroPortrait.execute(
        document_text=HERO_XML, background="Belebter Marktplatz in Gareth, warmes Tageslicht.",
        prompt_format=prompt_format,
    ).result
    assert request.call_count == 2
    assert anchor + separator + elaboration in result[0]
    assert "Gareth" in result[0]
    assert "warm daylight" in result[0]
    assert "Ollama-Ausarbeitung für 1 Textfelder übernommen." in result[3]
    assert "Ausarbeitung nicht übernommen" not in result[3]


@pytest.mark.parametrize("prompt_format", ["Natürliche Sprache", "Tags"])
def test_identical_expansion_does_not_duplicate_the_source_anchor(prompt_format, transport):
    request, overrides = transport
    anchor = "Warm daylight across the bustling market in Gareth"
    overrides["background"] = anchor + "."
    request.expansion_overrides["background"] = "  " + anchor + ".  "
    result = portrait.OnekoOllamaHeroPortrait.execute(
        document_text=HERO_XML, background="Warmes Tageslicht über dem belebten Markt in Gareth.",
        prompt_format=prompt_format,
    ).result
    assert request.call_count == 2
    assert result[0].count(anchor) == 1
    assert "Ollama-Ausarbeitung für 1 Textfelder übernommen." in result[3]
