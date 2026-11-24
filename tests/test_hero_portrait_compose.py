from copy import deepcopy

import pytest

from Oneko_ComfyUI_Nodes.nodes.hero_portrait_compose import build_portrait_plan, compile_portrait


def hero(species="Amaunir"):
    return {
        "source_format": "xml", "species": species, "gender": "männlich",
        "appearance": {"age": "20", "hair_color": "bronzefarben", "eye_color": "gelb"},
        "visible_traits": [
            "Natürlicher Rüstungsschutz: 1", "Zusätzliche Gliedmaßen: Balanceschwanz",
            "Springgelenke: 3", "Natürliche Waffen: Krallen",
        ],
        "equipment": ["Hemd", "Ring"],
    }


def task(**choices):
    return {"output_language": "English", **{key: {"selection": value} for key, value in choices.items()}}


def test_known_feline_species_gets_anatomy_hint_without_inventing_fur_color():
    controls = task(framing="Ganzer Körper")
    plan = build_portrait_plan(hero(), controls)
    assert plan["fields"]["hair"] == "Haarfarbe: bronzefarben"
    assert plan["fields"]["eyes"] == "Augenfarbe: gelb"
    assert "catlike face and fur" in plan["species_hint"]["English"]
    positive, _, _ = compile_portrait(plan, plan["fields"], [], "", controls)
    assert "bronze-colored hair" in positive
    assert "bronze fur" not in positive.casefold()
    assert "copper fur" not in positive.casefold()


def test_unknown_species_does_not_gain_a_guessed_anatomy():
    plan = build_portrait_plan(hero("Unbekannte Spezies"), task())
    assert plan["known_species"] is False
    assert plan["species_hint"] == {}


def test_explicit_age_override_can_omit_age_without_removing_species():
    source = hero("Thorwaler")
    controls = task()
    controls["additional_details"] = "Die Figur mit 60 Jahren darstellen."
    plan = build_portrait_plan(source, controls)
    localized = dict(plan["fields"], identity="A male Thorwaler", additional_details="Depicted at 60 years old")
    positive, _, _ = compile_portrait(plan, localized, [], "", controls, omitted_fact_ids=["age"])
    assert "20" not in positive
    assert "60 years old" in positive and "male Thorwaler" in positive


def test_unstructured_age_and_measurements_still_require_translation():
    source = hero("Unknown")
    source["appearance"] = {"age": "unbekannt", "height_cm": "ungefähr 180", "weight_kg": "unbekannt"}
    plan = build_portrait_plan(source, task())
    assert "unbekannt" in plan["fields"]["identity"]
    assert "age" not in plan["fields"]
    assert "identity" not in plan["fixed_translations"]
    assert "build" not in plan["fixed_translations"]


def test_firnelf_hint_preserves_explicit_black_hair_and_original_facts():
    source = hero("Firnelf")
    source["appearance"] = {"age": "30", "hair_color": "schwarz", "eye_color": "eisgrau"}
    original = deepcopy(source)
    controls = task(framing="Bis zum Bauch")
    plan = build_portrait_plan(source, controls)
    localized = dict(plan["fields"], identity="A 30-year-old male Firnelf", eyes="Ice-gray eyes", hair="Black hair")
    positive, _, _ = compile_portrait(plan, localized, [], "", controls)
    assert "black hair" in positive.casefold()
    assert "ice-gray eyes" in positive.casefold() and "30-year-old" in positive
    assert "pointed ears" in positive
    assert "white hair" not in positive.casefold() and "weiß" not in positive.casefold()
    assert source == original
    assert plan["fields"]["hair"] == "Haarfarbe: schwarz"
    assert plan["fields"]["eyes"] == "Augenfarbe: eisgrau"


def test_canonical_literals_replace_wrong_hair_and_inventory_translations():
    source = hero("Firnelf")
    source["appearance"]["hair_color"] = "schwarz"
    source["equipment"] = ["Armreif", "Federhut mit breiter Krempe"]
    original = deepcopy(source)
    controls = task(framing="Bis zum Bauch")
    plan = build_portrait_plan(source, controls)
    localized = dict(plan["fields"], hair="white hair", equipment_0="wedding ring", equipment_1="plain hat")
    positive, _, _ = compile_portrait(plan, localized, plan["equipment_ids"], "", controls)
    assert "black hair" in positive
    assert "bracelet" in positive
    assert "wide-brimmed feathered hat" in positive
    assert "white hair" not in positive and "wedding ring" not in positive and "plain hat" not in positive
    assert source == original
    assert plan["fields"]["hair"] == "Haarfarbe: schwarz"
    assert plan["fields"]["equipment_0"] == "Armreif"
    assert localized["hair"] == "white hair"


def test_unknown_compound_colors_and_items_keep_model_translation_without_guesses():
    source = hero()
    source["appearance"]["hair_color"] = "mitternachtsviolett irisierend"
    source["equipment"] = ["Sternenspiegel", "Armreif des Mondes"]
    controls = task(framing="Bis zum Bauch")
    plan = build_portrait_plan(source, controls)
    assert all(key not in plan["fixed_translations"] for key in ("hair", "equipment_0", "equipment_1"))
    localized = dict(plan["fields"], hair="Iridescent midnight-violet hair",
                     equipment_0="Star mirror", equipment_1="Moon bracelet")
    positive, _, _ = compile_portrait(plan, localized, plan["equipment_ids"], "", controls)
    assert all(value in positive for value in ("Iridescent midnight-violet hair", "Star mirror", "Moon bracelet"))


def test_numeric_armor_bonus_does_not_become_visible_armor_or_scales():
    plan = build_portrait_plan(hero(), task(framing="Ganzer Körper"))
    assert not any("Rüstungsschutz" in value for value in plan["fields"].values())
    assert any("Balanceschwanz" in value for value in plan["fields"].values())


@pytest.mark.parametrize("framing", ["Nur Kopf", "Kopf und Schultern", "Bis zum Bauch"])
def test_tail_and_leg_traits_are_excluded_from_close_crops(framing):
    plan = build_portrait_plan(hero(), task(framing=framing))
    assert not any("schwanz" in value.casefold() or "springgelenk" in value.casefold()
                   for value in plan["fields"].values())


@pytest.mark.parametrize("framing,visible", [
    ("Nur Kopf", False), ("Kopf und Schultern", False), ("Bis zum Bauch", True), ("Ganzer Körper", True),
])
def test_claws_are_kept_only_when_hands_can_be_in_the_crop(framing, visible):
    plan = build_portrait_plan(hero(), task(framing=framing))
    assert any("Krallen" in value for value in plan["fields"].values()) is visible


def test_no_equipment_mode_cannot_select_inventory_and_adds_simple_clothing():
    controls = task(equipment_mode="Ohne Ausrüstung")
    plan = build_portrait_plan(hero(), controls)
    assert plan["equipment_ids"] == []
    assert not any(key.startswith("equipment_") for key in plan["fields"])
    positive, _, _ = compile_portrait(plan, plan["fields"], [], "", controls)
    assert "Simple appropriate clothing" in positive
    assert "Ring" not in positive


def test_rear_view_does_not_require_visible_eyes_or_a_frontal_expression():
    controls = task(view="Von hinten", expression="Freundlich", framing="Ganzer Körper")
    plan = build_portrait_plan(hero(), controls)
    assert "eyes" not in plan["fields"]
    assert "face" not in plan["species_hint"]["English"]
    positive, _, _ = compile_portrait(plan, plan["fields"], [], "", controls)
    assert "Rear view" in positive
    assert "Friendly warm expression" not in positive


def test_omitting_identity_for_an_override_removes_its_species_hint():
    controls = task()
    plan = build_portrait_plan(hero(), controls)
    positive, _, report = compile_portrait(plan, plan["fields"], [], "", controls, omitted_fact_ids=["identity"])
    assert "catlike face" not in positive
    assert "Zusatzvorgaben" in report
