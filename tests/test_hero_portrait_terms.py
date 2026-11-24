import pytest

from Oneko_ComfyUI_Nodes.nodes.hero_portrait_terms import parse_term_explanations, resolve_term


def test_user_explanations_parse_comments_and_normalize_exact_keys():
    explanations = parse_term_explanations(
        "\ufeff# Eigene Begriffe\n\n  MITTELLA\u0308NDERIN  = Menschliche Reisende\n"
        "Hexe:  Eigene Schule = Beschreibung mit = im Text\n"
    )
    assert explanations == {
        "mittelländerin": "Menschliche Reisende",
        "hexe: eigene schule": "Beschreibung mit = im Text",
    }


@pytest.mark.parametrize("text,message", [
    ("# Kommentar\nFalsche Zeile", "Zeile 2"),
    ("= Erklärung", "Begriff und Erklärung"),
    ("Begriff = ", "Begriff und Erklärung"),
    ("Hexe = Eine Frau\n HEXE = Eine andere Frau", "mehrfach"),
    ("Äther = Erstes\nA\u0308ther = Zweites", "mehrfach"),
])
def test_invalid_or_duplicate_explanation_lines_are_rejected(text, message):
    with pytest.raises(ValueError, match=message):
        parse_term_explanations(text)


@pytest.mark.parametrize("value,kind,source,english", [
    ("Mittelländerin", "species", "Mensch", "human"),
    ("Mittelländer", "species", "Mensch", "human"),
    ("Mittella\u0308nderin", "species", "Mensch", "human"),
    ("Hexe", "profession", "Hexe", "witch"),
    ("Hexe: Schwester des Wissens", "profession", "Hexe", "witch"),
    ("Hexe: Verschwiegene Schwester", "profession", "Hexe", "witch"),
    ("Brusttuch", "equipment", "Tuch zur Bedeckung der Brust", "cloth chest covering"),
    ("Wasserschlauch", "equipment", "Trinkbehälter aus Leder", "waterskin"),
    ("Mittelländische Landbevölkerung - Fern der Zivilisation", "culture",
     "aus einer abgelegenen ländlichen Gemeinschaft", "from a remote rural community"),
])
def test_verified_complete_labels_get_plain_localized_descriptions(value, kind, source, english):
    en = resolve_term(value, kind)
    de = resolve_term(value, kind, language="German")
    assert en["source"] == de["source"] == source
    assert en["translation"] == english
    assert de["translation"] == source
    assert not en["unresolved"]
    assert set(en) == {"source", "translation", "note", "unresolved"}


@pytest.mark.parametrize("value,kind", [
    ("Halb-Mittelländerin", "species"),
    ("Eine Mittelländerin im roten Kleid", "species"),
    ("Mittelländerin", "equipment"),
    ("Hexe", "species"),
    ("Krieger: Unbekannte Schule", "profession"),
    ("Hexen: Fremde Schule", "profession"),
    ("Gueldenschlange", "equipment"),
    ("Wasserschlauch (7 Schank)", "equipment"),
])
def test_unknown_or_wrong_category_values_are_not_rewritten(value, kind):
    assert resolve_term(value, kind) == {
        "source": value, "translation": None, "note": "", "unresolved": "",
    }


def test_unknown_witch_specialization_uses_only_known_base_and_reports_omission():
    result = resolve_term("Hexe: Schule des Sternenfeuers", "profession")
    assert result["source"] == "Hexe"
    assert result["translation"] == "witch"
    assert "Schule des Sternenfeuers" in result["unresolved"]
    assert "Grundberuf" in result["unresolved"]
    assert "Schule des Sternenfeuers" in result["note"]


@pytest.mark.parametrize("value,kind", [
    ("Mittelländerin", "species"),
    ("Hexe: Schwester des Wissens", "profession"),
    ("Hexe: Schule des Sternenfeuers", "profession"),
    ("Unbekanntes Kleidungsstück", "equipment"),
])
def test_exact_user_override_wins_and_remains_available_for_ollama_localization(value, kind):
    description = "Eine eigene konkrete Beschreibung"
    overrides = parse_term_explanations(f"{value.upper()} = {description}")
    result = resolve_term(value, kind, overrides=overrides)
    assert result == {
        "source": description, "translation": None,
        "note": f"{value} → {description}", "unresolved": "",
    }


def test_user_override_never_replaces_substrings_or_unknown_subgroup_base():
    overrides = parse_term_explanations("Hexe = Kräuterkundige Reisende")
    assert resolve_term("Hexe: Unbekannt", "profession", overrides=overrides)["source"] == "Hexe"
    assert resolve_term("Die Hexe auf der Wiese", "profession", overrides=overrides)["source"] == "Die Hexe auf der Wiese"


@pytest.mark.parametrize("value", ["", "  Unbekanntes Wort  ", "Brusttuch und Kleid"])
def test_unknown_reference_text_is_preserved_exactly(value):
    assert resolve_term(value, "equipment")["source"] == value


def test_blank_custom_explanations_are_optional():
    assert parse_term_explanations(None) == {}
    assert parse_term_explanations("\n# Nur ein Kommentar\n") == {}


def test_invalid_call_parameters_do_not_silently_choose_a_language_or_category():
    with pytest.raises(ValueError, match="Begriffskategorie"):
        resolve_term("Hexe", "unknown")
    with pytest.raises(ValueError, match="Sprache"):
        resolve_term("Hexe", "profession", language="French")


@pytest.mark.parametrize("value,source,translated", [
    ("Tuchbeutel (5 Stein)", "Tuchbeutel", "cloth bag"),
    ("Tuchbeutel, 5 Stein", "Tuchbeutel", "cloth bag"),
    ("Wasserschlauch (5 Schank)", "Trinkbehälter aus Leder", "waterskin"),
    ("Wasserschlauch, 5 Schank", "Trinkbehälter aus Leder", "waterskin"),
])
def test_known_catalog_capacity_is_not_a_stone_inventory_or_visible_object_count(value, source, translated):
    result = resolve_term(value, "equipment")
    assert result["source"] == source
    assert result["translation"] == translated
    assert not any(character.isdigit() for character in result["source"])
    assert value in result["note"]
    assert resolve_term(value, "equipment", language="German")["translation"] == source


@pytest.mark.parametrize("value", [
    "Beutel mit 5 Steinen", "Tuchbeutel mit 5 Steinen", "Tuchbeutel (10 Stein)",
    "Unbekannter Behälter (5 Stein)", "Wasserschlauch (5 Schank) am Gürtel",
])
def test_capacity_handling_does_not_rewrite_literal_contents_or_unlisted_labels(value):
    assert resolve_term(value, "equipment") == {
        "source": value, "translation": None, "note": "", "unresolved": "",
    }


@pytest.mark.parametrize("value", ["Tuchbeutel (5 Stein)", "Wasserschlauch (5 Schank)"])
def test_custom_complete_catalog_label_overrides_builtin_capacity_normalization(value):
    overrides = parse_term_explanations(f"{value} = Mein ausdrücklich mit 5 Steinen gefüllter Beutel")
    result = resolve_term(value, "equipment", overrides=overrides)
    assert result["source"] == "Mein ausdrücklich mit 5 Steinen gefüllter Beutel"
    assert result["translation"] is None
