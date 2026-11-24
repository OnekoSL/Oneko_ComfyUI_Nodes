import json
import unicodedata
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import nodes as core_nodes
from PIL import Image
from comfy_extras.nodes_string import StringConcatenate

from Oneko_ComfyUI_Nodes.nodes import hero_document as document
from Oneko_ComfyUI_Nodes.nodes import ollama_hero_portrait as portrait
from Oneko_ComfyUI_Nodes.nodes.diffusion_clip_vae_cycler_loader import _filename_model_name


@pytest.fixture
def collection_path(tmp_path):
    path = tmp_path / "Heldensammlung.xml"
    path.write_text(
        '<helden><held name="Firuna Ehrwald"/><held name="Jäger Ährenfeld"/></helden>',
        encoding="utf-8",
    )
    return path


def test_selected_xml_hero_names_are_independent_of_collection_filename(collection_path):
    for hero_name, expected in (("Firuna Ehrwald", "Firuna_Ehrwald"), ("Jäger Ährenfeld", "Jäger_Ährenfeld")):
        hero = document.load_hero_document(document_path=collection_path, hero_name=hero_name)
        assert document.hero_filename_name(hero, document_path=collection_path) == expected


def test_in_place_document_change_changes_filename_name(tmp_path):
    path = tmp_path / "currently_selected.xml"
    names = []
    fingerprints = []
    for name in ("Alrik", "Firuna Ehrwald"):
        path.write_text(f'<held name="{name}"/>', encoding="utf-8")
        hero = document.load_hero_document(document_path=path)
        names.append(document.hero_filename_name(hero, document_path=path))
        fingerprints.append(document.hero_document_fingerprint(document_path=path, document_text=" \ufeff \n"))
    assert names == ["Alrik", "Firuna_Ehrwald"]
    assert fingerprints[0] != fingerprints[1]


@pytest.mark.parametrize("suffix,source", [("xml", "<held/>"), ("txt", "Eine Reisende mit grünem Mantel.")])
@pytest.mark.parametrize("inline", ["", " \n\t", " \ufeff \n"])
def test_unnamed_file_uses_actual_source_stem_with_quotes_and_bom(tmp_path, suffix, source, inline):
    path = tmp_path / f"3 Märchen Held.{suffix}"
    path.write_text(source, encoding="utf-8")
    quoted = f'  "{path}"  '
    hero = document.load_hero_document(document_path=quoted, document_text=inline)
    assert document.hero_filename_name(hero, document_path=quoted, document_text=inline) == "3_Märchen_Held"


@pytest.mark.parametrize("inline,expected", [
    ("Eine unbekannte Heldin.", "Unbenannter_Held"),
    (" \ufeff \nEine unbekannte Heldin.\n", "Unbenannter_Held"),
    ("<held/>", "Unbenannter_Held"),
    (' \ufeff <held name="Neuer Held"/> ', "Neuer_Held"),
])
def test_inline_document_never_inherits_old_file_name(tmp_path, inline, expected):
    stale_path = tmp_path / "Alter Held.xml"
    hero = document.load_hero_document(document_path=stale_path, document_text=inline)
    assert document.hero_filename_name(hero, document_path=stale_path, document_text=inline) == expected


@pytest.mark.parametrize("name,expected", [
    ("Jäger A\u0308hrenfeld", "Jäger_Ährenfeld"),
    (" .._ Firuna  Ehrwald_ . ", "Firuna_Ehrwald"),
    ('Lord<>:"/\\|?*%\x00\x1f\x7f\x85Alrik', "Lord_Alrik"),
    ("Held\tmit\nLeerraum\u00a0hier", "Held_mit_Leerraum_hier"),
    ("CON", "_CON"),
    ("prn.txt", "_prn.txt"),
    ("AUX", "_AUX"),
    ("NUL", "_NUL"),
    ("COM1", "_COM1"),
    ("LPT9.xml", "_LPT9.xml"),
    ("COM¹", "_COM¹"),
    ("LPT³", "_LPT³"),
    ("CONrad", "CONrad"),
    ("COM10", "COM10"),
    ("...___%", "Unbenannter_Held"),
])
def test_filename_token_is_windows_safe_and_preserves_unicode(name, expected):
    token = document.hero_filename_name({"name": name})
    assert token == expected
    assert unicodedata.is_normalized("NFC", token)


@pytest.mark.parametrize("name", ["Ä" * 140, "Alrik" * 20 + " .", "CON." + "a" * 94 + ".z"])
def test_long_names_stay_bounded_and_do_not_end_in_a_dot_or_space(name):
    token = document.hero_filename_name({"name": name})
    assert 0 < len(token) <= 100
    assert not token.endswith((".", " "))


def test_fifth_output_uses_selected_source_hero_not_ollama_name(collection_path):
    def fake_response(*args, **kwargs):
        payload = json.loads(args[2])
        translations = dict(payload["source_fields"])
        translations.update(payload.get("canonical_translations", {}))
        return json.dumps({
            "translations": translations, "equipment_ids": [], "omitted_fact_ids": [],
            "embellishment": "", "report": "Der Held heißt Falscher Modellname.",
        }, ensure_ascii=False)

    with patch.object(portrait, "_request_ollama", side_effect=fake_response) as request, \
         patch.object(portrait, "_unload_after_run"):
        for hero_name, expected in (("Firuna Ehrwald", "Firuna_Ehrwald"), ("Jäger Ährenfeld", "Jäger_Ährenfeld")):
            result = portrait.OnekoOllamaHeroPortrait.execute(
                document_path=collection_path, hero_name=hero_name,
                output_language="Deutsch", ollama_model="translator-only:latest",
            ).result
            assert len(result) == 5
            assert json.loads(result[2])["name"] == hero_name
            assert result[4] == expected
        assert request.call_count == 2


@pytest.mark.parametrize("hero_name,expected_token", [
    ("Jäger %width% %batch_num%", "Jäger_width_batch_num"),
    ("Raluf 'der Kühne'", "Raluf_'der_Kühne'"),
])
def test_core_concatenate_and_save_image_keep_hero_model_names_and_counter(tmp_path, hero_name, expected_token):
    hero_token = document.hero_filename_name({"name": hero_name})
    model_token = _filename_model_name("ANIMA/chiaroscuro moss v10.safetensors")
    joined = StringConcatenate.execute(hero_token, model_token, "__").result[0]
    assert joined == f"{expected_token}__chiaroscuro_moss_v10"
    prefix = StringConcatenate.execute("Heldenporträts/", joined, "").result[0]
    saver = core_nodes.SaveImage()
    saver.output_dir = str(tmp_path)
    images = torch.zeros((1, 3, 2, 3), device="cpu")

    for counter in (1, 2):
        saved = saver.save_images(images, filename_prefix=prefix)["ui"]["images"][0]
        assert saved["filename"] == f"{joined}_{counter:05}_.png"
        assert saved["subfolder"] == "Heldenporträts"
        image_path = tmp_path / saved["subfolder"] / saved["filename"]
        assert image_path.is_file()
        assert Path(saved["filename"]).name == saved["filename"]
        with Image.open(image_path) as image:
            assert image.size == (2, 3)
            assert image.mode == "RGB"
