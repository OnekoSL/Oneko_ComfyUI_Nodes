"""Resolve a small set of setting terms to plain portrait reference descriptions.

Only complete, typed source values are matched. These descriptions are data for
the portrait planner, never a replacement pass over the user's visual brief.
"""

import unicodedata


def _normalized(value):
    return " ".join(unicodedata.normalize("NFC", str(value or "")).split()).casefold()


# Mittelländer are human; neither complexion nor costume is implied here.
# https://dsa.ulisses-regelwiki.de/Spez_Menschen.html
_SPECIES = {
    "mittelländer": ("Mensch", "human"),
    "mittelländerin": ("Mensch", "human"),
    "mensch": ("Mensch", "human"),
    "menschen": ("Mensch", "human"),
}

# These are witch sisterhoods, not literal sisters/nuns or species. Their
# animal associations do not establish that a particular hero owns an animal.
# https://dsa.ulisses-regelwiki.de/ghfhgr.html
# https://dsa.ulisses-regelwiki.de/dffbfghgfh.html
_PROFESSIONS = {
    "hexe": ("Hexe", "witch"),
    "hexe: schwester des wissens": ("Hexe", "witch"),
    "hexe: verschwiegene schwester": ("Hexe", "witch"),
}

# The first source establishes a clothing item without a specific cut. The
# second establishes a drinking container, not a hose or a modern bottle.
# https://dsa.ulisses-regelwiki.de/RK_Ausruestung/kleidung/brusttuch.html
# https://dsa.ulisses-regelwiki.de/RK_Ausr_Wasserschlauch.html
# https://dsa.ulisses-regelwiki.de/Herstellung/die-herstellung-von-gewoehnlichen-gegenstaenden/herstellungstabelle-lederbearbeitung.html
_EQUIPMENT = {
    "brusttuch": ("Tuch zur Bedeckung der Brust", "cloth chest covering"),
    "wasserschlauch": ("Trinkbehälter aus Leder", "waterskin"),
    "tuchbeutel": ("Tuchbeutel", "cloth bag"),
    # These complete catalog labels describe capacity, not visible contents.
    # Keep the original label in hero_reference_data; leave its bookkeeping
    # quantity out of the visual source field and canonical translation.
    # Stein is a weight unit (the cited bag can hold a weight of 10 Stein):
    # https://dsa.ulisses-regelwiki.de/traditionsartefakt.html?traditionsartefakt=schuppenbeutel
    # Schank measures liquid volume (1 Schank = 250 ml):
    # https://dsa.ulisses-regelwiki.de/zauberrune.html?zauberrune=Salzwasserrune%20(Sj%C3%B6vannruna)
    "tuchbeutel (5 stein)": ("Tuchbeutel", "cloth bag"),
    "tuchbeutel, 5 stein": ("Tuchbeutel", "cloth bag"),
    "wasserschlauch (5 schank)": ("Trinkbehälter aus Leder", "waterskin"),
    "wasserschlauch, 5 schank": ("Trinkbehälter aus Leder", "waterskin"),
}

# This paraphrases the explicit source label, without inferring an occupation,
# complexion, outfit, or the geography of the requested portrait background.
_CULTURES = {
    "mittelländische landbevölkerung - fern der zivilisation": (
        "aus einer abgelegenen ländlichen Gemeinschaft", "from a remote rural community",
    ),
}
_TERMS = {
    "species": _SPECIES, "profession": _PROFESSIONS,
    "equipment": _EQUIPMENT, "culture": _CULTURES,
}


def parse_term_explanations(text):
    """Parse explicit ``Begriff = Erklärung`` lines; reject ambiguous entries."""
    explanations = {}
    for line_number, raw in enumerate(str(text or "").lstrip("\ufeff").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Begriffserklärung in Zeile {line_number}: Bitte 'Begriff = Erklärung' angeben.")
        term, description = (part.strip() for part in line.split("=", 1))
        key = _normalized(term)
        if not key or not description:
            raise ValueError(f"Begriffserklärung in Zeile {line_number}: Begriff und Erklärung dürfen nicht leer sein.")
        if key in explanations:
            raise ValueError(f"Begriffserklärung in Zeile {line_number}: {term!r} ist mehrfach definiert.")
        explanations[key] = description
    return explanations


def resolve_term(value, kind, language="English", overrides=None):
    """Return plain reference data and, when known, a canonical translation.

    Unknown labels pass through unchanged. Only an unknown colon-separated
    specialization of a known profession falls back to its verified base.
    User explanations override exact complete labels and still need Ollama's
    localization, so their ``translation`` is deliberately ``None``.
    """
    if kind not in _TERMS:
        raise ValueError(f"Unbekannte Begriffskategorie: {kind!r}")
    if language not in ("English", "German"):
        raise ValueError(f"Unbekannte Sprache für Begriffserklärungen: {language!r}")
    original = str(value or "")
    key = _normalized(original)
    result = {"source": original, "translation": None, "note": "", "unresolved": ""}
    if overrides and key in overrides:
        description = overrides[key]
        result.update(source=description, note=f"{original} → {description}")
        return result

    entry = _TERMS[kind].get(key)
    if entry is None and kind == "profession" and ":" in key:
        base = key.split(":", 1)[0].strip()
        entry = _PROFESSIONS.get(base)
        if entry is not None:
            result["unresolved"] = (
                f"Unbekannte Spezialisierung {original!r}; nur der Grundberuf {entry[0]!r} wird beschrieben."
            )
    if entry is not None:
        source, translated = entry
        result.update(
            source=source,
            translation=translated if language == "English" else source,
            note=f"{original} → {source}" if original != source else "",
        )
    return result
