"""Build a source-indexed portrait brief and compose its localized parts locally.

The language model localizes individual source facts; it does not decide whether
the user's background or the selected framing survives into the final prompt.
"""

import re
import unicodedata

from .hero_portrait_terms import parse_term_explanations, resolve_term


# Conservative anatomical descriptions, separate from the imported hero data.
# Amaunir: https://uhrwerk-verlag.de/myranor/
# Elves: https://dsa.ulisses-regelwiki.de/Spez_Elfen.html
_CAT_PERSON = {
    "English": "An anthropomorphic feline person with a clearly catlike face and fur",
    "German": "Eine anthropomorphe Katzenperson mit deutlich katzenartigem Gesicht und Fell",
}
_ELF = {
    "English": "An elf with distinctly pointed ears",
    "German": "Eine elfische Person mit deutlich spitzen Ohren",
}
_SPECIES_HINTS = {
    "amaunir": _CAT_PERSON,
    **{name: _ELF for name in ("elf", "elfe", "elfen", "firnelf", "firnelfe", "auelf", "auelfe", "waldelf", "waldelfe")},
}

# Exact source labels only: unfamiliar colors and items remain model-localized.
# No substring replacement is applied to free text, compound inventory names or
# appearance descriptions, so this table cannot silently add unrecorded details.
_COLORS_EN = {
    "schwarz": "black", "weiß": "white", "weiss": "white", "grau": "gray",
    "hellgrau": "light gray", "dunkelgrau": "dark gray", "eisgrau": "ice-gray",
    "silber": "silver", "silbern": "silver", "silberfarben": "silver-colored",
    "silberweiß": "silvery white", "silberweiss": "silvery white",
    "blau": "blue", "hellblau": "light blue", "dunkelblau": "dark blue",
    "eisblau": "ice-blue", "himmelblau": "sky blue", "graublau": "gray-blue",
    "blaugrau": "blue-gray", "grün": "green", "hellgrün": "light green",
    "dunkelgrün": "dark green", "graugrün": "gray-green", "grüngrau": "green-gray",
    "braun": "brown", "hellbraun": "light brown", "dunkelbraun": "dark brown",
    "schwarzbraun": "black-brown", "rotbraun": "reddish-brown", "kastanienbraun": "chestnut-brown",
    "blond": "blond", "hellblond": "light blond", "dunkelblond": "dark blond",
    "aschblond": "ash-blond", "platinblond": "platinum-blond", "goldblond": "golden blond",
    "rot": "red", "hellrot": "light red", "dunkelrot": "dark red", "kirschrot": "cherry-red",
    "rötlich": "reddish", "gelb": "yellow", "gold": "gold", "golden": "golden",
    "goldfarben": "gold-colored", "bernstein": "amber", "bernsteinfarben": "amber",
    "bronze": "bronze", "bronzefarben": "bronze-colored",
    "kupfer": "copper", "kupferfarben": "copper-colored", "kupferrot": "copper-red",
    "violett": "violet", "lila": "purple", "rosa": "pink", "türkis": "turquoise",
}
_TRAITS_EN = {
    "krallen": "claws",
    "balanceschwanz": "balancing tail",
    "schwanz": "tail",
}
_EQUIPMENT_EN = {
    "armreif": "bracelet",
    "brotbeutel": "bread bag",
    "dolch": "dagger",
    "dolchscheide": "dagger sheath",
    "federhut": "feathered hat",
    "federhut mit breiter krempe": "wide-brimmed feathered hat",
    "gürteltasche": "belt pouch",
    "gürteltasche, leder": "leather belt pouch",
    "gürteltasche, hartleder": "hard-leather belt pouch",
    "hose": "trousers",
    "ledergürtel": "leather belt",
    "ohrring": "earring",
    "hemd": "shirt",
    "halskette": "necklace",
    "kleid": "dress",
    "lederweste": "leather vest",
    "rock": "skirt",
    "wolldecke": "wool blanket",
    "ring": "ring",
    "tunika, leinen": "linen tunic",
    "leinentunika": "linen tunic",
    "leichte stiefel": "light boots",
    "steinmesser": "stone knife",
    "wurfspeer": "javelin",
}

# These are descriptive prompt fragments, not instructions sent to a model.
_FRAMING = {
    "Nur Kopf": (
        "Head-only close-up, with the entire head, ears and headwear visible, cropped at the neck",
        "Kopfporträt als Nahaufnahme mit vollständig sichtbarem Kopf, Ohren und Kopfbedeckung, Bildausschnitt bis zum Hals",
    ),
    "Kopf und Schultern": (
        "Head-and-shoulders portrait, with the entire head, ears and headwear visible, cropped at the upper chest",
        "Kopf-Schulter-Porträt mit vollständig sichtbarem Kopf, Ohren und Kopfbedeckung, Bildausschnitt bis zur oberen Brust",
    ),
    "Bis zum Bauch": (
        "Waist-up portrait, framed from the entire head to the belly and waist",
        "Porträt bis zum Bauch, Bildausschnitt vom vollständigen Kopf bis zu Bauch und Taille",
    ),
    "Dreiviertel (bis Knie)": (
        "Three-quarter-length portrait, framed from the entire head to the knees",
        "Dreiviertelporträt, Bildausschnitt vom vollständigen Kopf bis zu den Knien",
    ),
    "Ganzer Körper": (
        "Full-body portrait, the entire head and both feet fully visible, with space around the silhouette",
        "Ganzkörperporträt mit vollständig sichtbarem Kopf und beiden Füßen sowie Abstand um die Silhouette",
    ),
}
_VIEW = {
    "Dreiviertelansicht": ("Three-quarter view at eye level", "Dreiviertelansicht auf Augenhöhe"),
    "Frontal": ("Frontal view at eye level", "Frontalansicht auf Augenhöhe"),
    "Profil": ("Side profile at eye level", "Seitenprofil auf Augenhöhe"),
    "Von hinten": ("Rear view of the character", "Rückansicht der Figur"),
    "Leichte Untersicht": ("Subtle low-angle view", "Leichte Untersicht"),
    "Leichte Aufsicht": ("Subtle high-angle view", "Leichte Aufsicht"),
}
_ART_STYLE = {
    "Fantasy-Illustration": (
        "Painterly fantasy book illustration with visible brush texture, natural proportions and facial structure appropriate to the character's species and age, believable material textures and softly modeled volumes",
        "Malerische Fantasy-Buchillustration mit sichtbarer Pinselstruktur, natürlichen Proportionen und einer zur Spezies und zum Alter passenden Gesichtsstruktur, glaubwürdigen Materialtexturen und weich modellierten Formen",
    ),
    "Realistische Malerei": (
        "Realistic painted portrait with subtle brushwork, natural proportions appropriate to the character's species and age, lifelike material textures and carefully modeled light and shadow",
        "Realistisch gemaltes Porträt mit dezentem Pinselstrich, natürlichen zur Spezies und zum Alter passenden Proportionen, naturgetreuen Materialtexturen und sorgfältig modelliertem Licht und Schatten",
    ),
    "Fotorealistisch": (
        "Photorealistic portrait with photographic lighting, fine natural surface detail and physically plausible materials, preserving the character's stated species, anatomy and age",
        "Fotorealistisches Porträt mit fotografischer Lichtführung, feinen natürlichen Oberflächendetails und physikalisch glaubwürdigen Materialien unter Beibehaltung der angegebenen Spezies, Anatomie und des Alters",
    ),
    "Ölgemälde": (
        "Traditional oil painting with rich layered pigments, visible brushwork and canvas texture, naturally modeled forms appropriate to the character's species and age",
        "Traditionelles Ölgemälde mit satten geschichteten Pigmenten, sichtbarem Pinselstrich und Leinwandstruktur sowie natürlich modellierten, zur Spezies und zum Alter passenden Formen",
    ),
    "Aquarell": (
        "Watercolor portrait with translucent pigment washes, soft color bleeding, delicate painted edges and visible textured paper",
        "Aquarellporträt mit durchscheinenden Pigmentlasuren, sanft verlaufenden Farben, zarten gemalten Kanten und sichtbar strukturiertem Papier",
    ),
    "Anime": (
        "Anime character illustration with expressive stylized features, clean line art and cel shading, preserving the character's species and stated age",
        "Anime-Charakterillustration mit ausdrucksstarken stilisierten Gesichtszügen, klaren Konturlinien und Cel-Shading unter Beibehaltung der Spezies und des angegebenen Alters",
    ),
    "Comic": (
        "Comic illustration with expressive ink lines, deliberate flat color shapes and graphic shadows, preserving the character's species and stated age",
        "Comicillustration mit ausdrucksstarken Tuschelinien, gezielten flächigen Farbformen und grafischen Schatten unter Beibehaltung der Spezies und des angegebenen Alters",
    ),
    "Eigene Vorgabe": ("", ""),
}
# Share the actual output wording with the Ollama brief, rather than maintaining
# a second, more detailed style instruction which never reaches the encoder.
ART_STYLES = {name: phrases[0] for name, phrases in _ART_STYLE.items()}
ART_STYLES["Eigene Vorgabe"] = (
    "Use the medium and style from additional_details; if unspecified, use a neutral character illustration."
)
_PAINTED_STYLE_NEGATIVE = (
    "anime, manga, chibi, cel shading, oversized stylized eyes, cartoon, glossy 3D render, plastic skin, photograph",
    "Anime, Manga, Chibi, Cel-Shading, übergroße stilisierte Augen, Cartoon, glänzendes 3D-Rendering, Plastikhaut, Fotografie",
)
_ART_STYLE_NEGATIVE = {
    **{name: _PAINTED_STYLE_NEGATIVE for name in ("Fantasy-Illustration", "Realistische Malerei", "Ölgemälde", "Aquarell")},
    "Fotorealistisch": (
        "painting, illustration, anime, manga, cartoon, line art, cel shading, 3D render, plastic skin",
        "Malerei, Illustration, Anime, Manga, Cartoon, Konturzeichnung, Cel-Shading, 3D-Rendering, Plastikhaut",
    ),
    "Anime": ("photograph, live action, oil painting, 3D render", "Fotografie, Realfilm, Ölgemälde, 3D-Rendering"),
    "Comic": ("photograph, live action, oil painting, 3D render", "Fotografie, Realfilm, Ölgemälde, 3D-Rendering"),
    "Eigene Vorgabe": ("", ""),
}
_EXPRESSION = {
    "Aus Held und Pose": ("", ""),
    "Neutral": ("Calm neutral expression", "Ruhiger neutraler Gesichtsausdruck"),
    "Freundlich": ("Friendly warm expression", "Freundlicher warmer Gesichtsausdruck"),
    "Ernst": ("Serious composed expression", "Ernster gefasster Gesichtsausdruck"),
    "Entschlossen": ("Determined confident expression", "Entschlossener selbstbewusster Gesichtsausdruck"),
    "Neugierig": ("Curious alert expression", "Neugieriger aufmerksamer Gesichtsausdruck"),
    "Geheimnisvoll": ("Subtle enigmatic expression", "Dezent geheimnisvoller Gesichtsausdruck"),
}
_LIGHTING = {
    "Passend zur Szene": ("", ""),
    "Weiches Tageslicht": ("Soft diffuse daylight", "Weiches diffuses Tageslicht"),
    "Goldene Stunde": ("Warm golden-hour light", "Warmes Licht der goldenen Stunde"),
    "Dramatisches Seitenlicht": ("Dramatic side lighting with controlled shadows", "Dramatisches Seitenlicht mit gezielten Schatten"),
    "Kerzenlicht": ("Warm candlelight with soft falloff", "Warmes Kerzenlicht mit sanftem Lichtabfall"),
    "Mondlicht": ("Cool moonlight with gentle reflected fill", "Kühles Mondlicht mit sanftem reflektiertem Aufhelllicht"),
    "Studiolicht": ("Soft studio key light with subtle rim lighting", "Weiches Studio-Hauptlicht mit dezentem Konturlicht"),
}


def _selection(task, name, default):
    return task.get(name, {}).get("selection", default)


def _text(value):
    return str(value or "").strip()


def _trait_source(value, framing):
    value = _text(value)
    folded = value.casefold()
    # A numerical rule bonus does not tell us whether a body has fur or scales.
    if re.fullmatch(r"natürlicher rüstungsschutz(?:\s*:\s*[+-]?\d+(?:[.,]\d+)?)?", folded):
        return ""
    close_crop = framing in ("Nur Kopf", "Kopf und Schultern", "Bis zum Bauch")
    if close_crop and ("schwanz" in folded or "springgelenk" in folded):
        return ""
    if framing in ("Nur Kopf", "Kopf und Schultern") and "krallen" in folded:
        return ""
    for prefix in ("natürliche waffen:", "zusätzliche gliedmaßen:", "natürlicher rüstungsschutz:"):
        if folded.startswith(prefix.casefold()):
            return value[len(prefix):].strip()
    return value


def build_portrait_plan(hero, task):
    """Index only source facts which are useful within the selected portrait crop."""
    fields = {}
    fixed_translations = {}
    character_ids = []
    equipment_ids = []
    term_notes = {}
    unresolved_terms = {}
    language = task.get("output_language", "English")
    overrides = parse_term_explanations(task.get("term_explanations", ""))

    def resolve(identifier, value, kind):
        resolved = resolve_term(value, kind, language, overrides)
        if resolved["note"]:
            term_notes[identifier] = resolved["note"]
        if resolved["unresolved"]:
            unresolved_terms[identifier] = resolved["unresolved"]
        return resolved

    def add(identifier, value, target=None):
        clean = _text(value)
        if clean:
            fields[identifier] = clean
            if target is not None:
                target.append(identifier)

    def fix(identifier, raw_value, glossary, suffix=""):
        translated = glossary.get(_text(raw_value).casefold())
        if identifier in fields and translated:
            fixed_translations[identifier] = (
                fields[identifier] if task.get("output_language", "English") == "German"
                else translated + suffix
            )

    appearance = hero.get("appearance", {})
    species = _text(hero.get("species"))
    hint = _SPECIES_HINTS.get(species.casefold(), {})
    custom_species = " ".join(unicodedata.normalize("NFC", species).split()).casefold() in overrides
    if custom_species:
        hint = {}
    rear_view = _selection(task, "view", "Dreiviertelansicht") == "Von hinten"
    if rear_view and species.casefold() == "amaunir" and not custom_species:
        hint = {
            "English": "An anthropomorphic feline person with fur",
            "German": "Eine anthropomorphe Katzenperson mit Fell",
        }
    if hero.get("source_format") == "text":
        add("description", hero.get("description"), character_ids)
    else:
        species_term = resolve("identity", species, "species")
        gender = _text(hero.get("gender"))
        age = _text(appearance.get("age"))
        numeric_age = bool(re.fullmatch(r"\d+(?:[.,]\d+)?", age))
        identity = [value for value in (species_term["source"], gender) if value]
        if age and not numeric_age:
            identity.append(f"{age} Jahre alt")
        add("identity", "; ".join(identity) or "Eine einzelne Fantasy-Figur", character_ids)
        # Preserve known species/gender/age locally; the original setting label
        # stays in character_summary rather than confusing the image encoder.
        genders = {"männlich": "male", "weiblich": "female", "male": "male", "female": "female", "": ""}
        if species_term["translation"] and gender.casefold() in genders:
            if language == "German":
                fixed_translations["identity"] = fields["identity"]
            else:
                identity_en = ["A " + " ".join(value for value in (genders[gender.casefold()], species_term["translation"]) if value)]
                if age and not numeric_age:
                    identity_en.append(f"{age} years old")
                fixed_translations["identity"] = ", ".join(value for value in identity_en if value)
        # Numeric age is already structured data. Keep it separate from unknown
        # species labels so a translation cannot omit it or spell it differently.
        if numeric_age:
            add("age", f"{age} Jahre alt", character_ids)
            fixed_translations["age"] = fields["age"] if language == "German" else f"{age} years old"
        for index, value in enumerate(hero.get("professions", [])):
            identifier = f"profession_{index}"
            resolved = resolve(identifier, value, "profession")
            add(identifier, resolved["source"], character_ids)
            if identifier in fields and resolved["translation"]:
                fixed_translations[identifier] = resolved["translation"]
        if not rear_view and _text(appearance.get("eye_color")):
            add("eyes", f"Augenfarbe: {appearance['eye_color']}", character_ids)
            fix("eyes", appearance["eye_color"], _COLORS_EN, " eyes")
        if _text(appearance.get("hair_color")):
            # This is hair, even for species with fur. No fur color is invented.
            add("hair", f"Haarfarbe: {appearance['hair_color']}", character_ids)
            fix("hair", appearance["hair_color"], _COLORS_EN, " hair")
        size = []
        if _text(appearance.get("height_cm")):
            size.append(f"{appearance['height_cm']} cm groß")
        if _text(appearance.get("weight_kg")):
            size.append(f"{appearance['weight_kg']} kg schwer")
        add("build", "; ".join(size), character_ids)
        measurements = [(appearance.get("height_cm"), "cm tall"), (appearance.get("weight_kg"), "kg")]
        present_measurements = [(str(value), unit) for value, unit in measurements if _text(value)]
        if present_measurements and all(re.fullmatch(r"\d+(?:[.,]\d+)?", value) for value, _ in present_measurements):
            fixed_translations["build"] = fields["build"] if language == "German" else "; ".join(
                f"{value} {unit}" for value, unit in present_measurements
            )
        for index, value in enumerate(hero.get("appearance_text", [])):
            add(f"appearance_{index}", value, character_ids)
        framing = _selection(task, "framing", "Bis zum Bauch")
        for index, value in enumerate(hero.get("visible_traits", [])):
            source = _trait_source(value, framing)
            add(f"trait_{index}", source, character_ids)
            fix(f"trait_{index}", source, _TRAITS_EN)
        if _selection(task, "equipment_mode", "Sichtbare Ausrüstung") != "Ohne Ausrüstung":
            for index, value in enumerate(hero.get("equipment", [])):
                identifier = f"equipment_{index}"
                resolved = resolve(identifier, value, "equipment")
                add(identifier, resolved["source"], equipment_ids)
                if identifier in fields and resolved["translation"]:
                    fixed_translations[identifier] = resolved["translation"]
                elif not resolved["note"]:
                    fix(identifier, value, _EQUIPMENT_EN)
    for name in ("pose", "background", "additional_details"):
        add(name, task.get(name))
    return {
        "fields": fields,
        "fixed_translations": fixed_translations,
        "character_ids": character_ids,
        "equipment_ids": equipment_ids,
        "known_species": bool(hint),
        "species_hint": dict(hint),
        "term_notes": term_notes,
        "unresolved_terms": unresolved_terms,
    }


def _join_fragments(parts, tags):
    parts = [_text(part).strip(" \t\r\n.,;") for part in parts if _text(part)]
    if tags:
        return ", ".join(parts)
    return " ".join(part if part.endswith(("!", "?")) else f"{part}." for part in parts)


def compile_portrait(plan, localized, selected_equipment, embellishment, task, omitted_fact_ids=()):
    """Compose validated localized sources, plus local framing and style controls.

    Brief fields are always retained. Omissions affect only document facts, and
    the caller is responsible for validating that additional_details permits them.
    """
    language = task.get("output_language", "English")
    language_index = 1 if language == "German" else 0
    localized = {**localized, **plan.get("fixed_translations", {})}
    omitted = set(omitted_fact_ids)
    selected = set(selected_equipment)
    used_facts = [key for key in plan["character_ids"] if key not in omitted]
    used_equipment = [key for key in plan["equipment_ids"] if key in selected and key not in omitted]
    style = _selection(task, "art_style", "Fantasy-Illustration")
    # Establish the visual medium before character/inventory details. These
    # descriptive phrases also work with encoders that ignore prompt weights.
    parts = [_ART_STYLE[style][language_index]]
    hint = plan.get("species_hint", {}).get(language, "") if "identity" not in omitted else ""
    for key in used_facts:
        parts.append(localized[key])
        if key == "identity" and hint:
            parts.append(hint)
    if used_equipment:
        label = "Kleidung und Ausrüstung" if language == "German" else "Clothing and equipment"
        parts.append(f"{label}: " + ", ".join(localized[key] for key in used_equipment))
    elif _selection(task, "equipment_mode", "Sichtbare Ausrüstung") == "Ohne Ausrüstung":
        parts.append("Schlichte angemessene Kleidung" if language == "German" else "Simple appropriate clothing")
    if "pose" in plan["fields"]:
        parts.append(localized["pose"])
    for name, options, default in (
        ("framing", _FRAMING, "Bis zum Bauch"),
        ("view", _VIEW, "Dreiviertelansicht"),
        ("expression", _EXPRESSION, "Aus Held und Pose"),
    ):
        if name == "expression" and _selection(task, "view", "Dreiviertelansicht") == "Von hinten":
            continue
        parts.append(options[_selection(task, name, default)][language_index])
    if "background" in plan["fields"]:
        parts.append(localized["background"])
    parts.append(_LIGHTING[_selection(task, "lighting", "Passend zur Szene")][language_index])
    if "additional_details" in plan["fields"]:
        parts.append(localized["additional_details"])
    elif style == "Eigene Vorgabe":
        parts.append("Charakterillustration" if language == "German" else "Character illustration")
    parts.append(embellishment)
    positive = _join_fragments(parts, _selection(task, "prompt_format", "Natürliche Sprache") == "Tags")
    negative = "mehrfach dargestellte Figur, Collage, Wasserzeichen" if language == "German" else "duplicate character, collage, watermark"
    if _selection(task, "framing", "Bis zum Bauch") == "Ganzer Körper":
        negative += ", abgeschnittene Füße" if language == "German" else ", cropped feet"
    style_negative = _ART_STYLE_NEGATIVE[style][language_index]
    if style_negative:
        negative += ", " + style_negative
    report = f"{len(used_facts)} Charakterangaben und {len(used_equipment)} Inventareinträge übernommen. Bildausschnitt und Blickwinkel lokal festgelegt. Bildstil: {style} (lokal)."
    if style_negative:
        report += " Stilbeschreibung am Promptanfang; passende Stilausschlüsse im Negativ-Prompt."
    if "pose" in plan["fields"]:
        report += " Pose übernommen."
    if "background" in plan["fields"]:
        report += " Hintergrund übernommen."
    if hint:
        report += " Feste Speziesbeschreibung ergänzt."
    if omitted:
        report += f" {len(omitted)} Dokumentangaben gemäß Zusatzvorgaben ausgelassen."
    report += f" Ausführlichkeit: {_selection(task, 'prompt_detail', 'Ausführlich')}."
    used_ids = set(used_facts + used_equipment)
    term_notes = [note for key, note in plan.get("term_notes", {}).items() if key in used_ids]
    unresolved = [note for key, note in plan.get("unresolved_terms", {}).items() if key in used_ids]
    if term_notes:
        report += " Begriffe aufgelöst: " + "; ".join(dict.fromkeys(term_notes)) + "."
    if unresolved:
        report += " Ungeklärte Fachbegriffe: " + "; ".join(dict.fromkeys(unresolved)) + "."
    return positive, negative, report
