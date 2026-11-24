"""Read portrait-relevant reference data from Helden Software XML or plain text."""

import hashlib
import json
import logging
import os
from pathlib import Path
import re
import stat
import unicodedata
import xml.etree.ElementTree as ET

import folder_paths


MAX_DOCUMENT_BYTES = 4 * 1024 * 1024
HERO_EXTENSIONS = frozenset((".xml", ".txt"))
HERO_ROOTS_CONFIG = "oneko_hero_document_paths.json"
_VISIBLE_TRAITS = (
    "aussehen", "albino", "fell", "schuppen", "haut", "kiemen", "flügel",
    "hörner", "schwanz", "gliedmaß", "krallen", "hauer", "wuchs", "narben",
    "verstümmelt", "einarmig", "einbeinig", "einäugig", "fettleibig", "dürr",
    "unansehnlich", "stigma", "natürliche waffen", "springgelenke",
    "natürlicher rüstungsschutz",
)


class _NoDTDTreeBuilder(ET.TreeBuilder):
    def doctype(self, name, pubid, system):
        raise ValueError("Helden-XML darf keine DTD- oder Entity-Deklarationen enthalten.")


def _clean(value):
    return " ".join(str(value or "").split())


def _tag(element):
    return element.tag.rsplit("}", 1)[-1]


def _children(element, name):
    return [] if element is None else [child for child in element if _tag(child) == name]


def _child(element, name):
    return next(iter(_children(element, name)), None)


def _attribute(element, name):
    return _clean(element.get(name)) if element is not None else ""


def _label(element):
    explicit = _attribute(element, "string")
    if explicit:
        return explicit
    value = _attribute(element, "name")
    return value.rsplit(".", 1)[-1] if value.startswith("helden.") else value


def _unique(values):
    return list(dict.fromkeys(value for value in values if value))


class HeroPathOutsideRoots(ValueError):
    def __init__(self):
        super().__init__("Der Pfad liegt außerhalb der freigegebenen Ordner für Heldendokumente.")


def hero_document_roots(input_directory=None):
    """Return the configured and resolved forms of existing, trusted roots."""
    if input_directory is None:
        input_directory = folder_paths.get_input_directory()
    config = Path(folder_paths.base_path) / HERO_ROOTS_CONFIG
    configured = []
    try:
        settings = json.loads(config.read_text(encoding="utf-8-sig"))
        if not isinstance(settings, dict) or not isinstance(settings.get("roots"), list):
            raise ValueError("The configuration must contain a roots array.")
        configured = [path for path in settings["roots"] if isinstance(path, str) and Path(path).is_absolute()]
    except FileNotFoundError:
        pass
    except (OSError, ValueError):
        logging.warning("Oneko Hero Portrait: %s could not be read. Only ComfyUI input is approved.", config.name)
    roots = {}
    for directory in (input_directory, *configured):
        try:
            root = Path(os.path.abspath(directory))
            resolved = root.resolve(strict=True)
            if resolved.is_dir():
                roots[root] = resolved
        except (OSError, RuntimeError, ValueError):
            continue
    return roots


def hero_path_text(value):
    value = str(value or "").strip()
    if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value


def parse_hero_path(value):
    if "\0" in value or re.match(r"^[a-z][a-z0-9+.-]*://", value, re.I):
        raise ValueError("Bitte einen Dateisystempfad statt einer URL angeben.")
    try:
        path = Path(value).expanduser()
    except RuntimeError as error:
        raise ValueError("Der Dateisystempfad ist ungültig.") from error
    if os.name == "nt" and (
        path.drive.startswith(("\\\\?\\", "\\\\.\\"))
        or (path.anchor and not path.is_absolute())
        or any(":" in part for part in path.parts[bool(path.anchor):])
        or path.is_reserved()
    ):
        raise ValueError("Der Windows-Dateisystempfad ist ungültig.")
    return path


def resolve_hero_path(path, roots):
    """Check the requested location, then check the real target before use."""
    lexical = Path(os.path.abspath(path))
    if not any(lexical.is_relative_to(root) for root in (*roots, *roots.values())):
        raise HeroPathOutsideRoots()
    resolved = path.resolve(strict=True)
    if not in_resolved_hero_roots(resolved, roots.values()):
        raise HeroPathOutsideRoots()
    return resolved


def in_resolved_hero_roots(path, roots):
    # Resolved spelling also preserves the boundary on case-sensitive NTFS directories.
    return any(path.parts[:len(root.parts)] == root.parts for root in roots)


def _read_file(document_path):
    value = hero_path_text(document_path)
    if not value:
        raise ValueError("Bitte einen Helden-Dateipfad oder Heldentext angeben.")
    path = parse_hero_path(value)
    if path.suffix.lower() not in HERO_EXTENSIONS:
        raise ValueError("Heldendokumente müssen eine .xml- oder .txt-Datei sein.")
    input_directory = Path(folder_paths.get_input_directory())
    roots = hero_document_roots(input_directory)
    if not path.is_absolute():
        # Preserve existing paths relative to the working directory when they
        # already point into an approved folder. Otherwise use ComfyUI input.
        from_working_directory = Path(os.path.abspath(path))
        if not any(from_working_directory.is_relative_to(root) for root in (*roots, *roots.values())):
            path = input_directory / path
    try:
        resolved = resolve_hero_path(path, roots)
        if resolved.suffix.lower() not in HERO_EXTENSIONS or not stat.S_ISREG(resolved.stat().st_mode):
            raise ValueError("Heldendokumente müssen reguläre .xml- oder .txt-Dateien sein.")
        with resolved.open("rb") as source:
            data = source.read(MAX_DOCUMENT_BYTES + 1)
    except (OSError, RuntimeError) as error:
        raise ValueError("Heldendokument kann nicht gelesen werden.") from error
    if len(data) > MAX_DOCUMENT_BYTES:
        raise ValueError("Das Heldendokument ist größer als 4 MiB.")
    return data, resolved.suffix.lower()


def _decode_text(data):
    if data.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        return data.decode("utf-32")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252")


def _xml_hero(data, hero_name):
    try:
        root = ET.fromstring(data, parser=ET.XMLParser(target=_NoDTDTreeBuilder()))
    except ET.ParseError as error:
        raise ValueError(f"Ungültiges Helden-XML: {error}") from error
    if _tag(root) == "held":
        heroes = [root]
    elif _tag(root) == "helden":
        heroes = _children(root, "held")
    else:
        raise ValueError("Das XML enthält kein Helden-Software-Dokument (<helden> oder <held>).")
    if not heroes:
        raise ValueError("Das Helden-XML enthält keinen Helden.")
    selected_name = str(hero_name or "").strip()
    if selected_name:
        matches = [hero for hero in heroes if _attribute(hero, "name") == selected_name]
        if len(matches) != 1:
            raise ValueError(f"Held {selected_name!r} ist nicht eindeutig vorhanden; bitte den exakten Namen angeben.")
        return matches[0]
    if len(heroes) != 1:
        names = ", ".join(_attribute(hero, "name") or "(ohne Namen)" for hero in heroes)
        raise ValueError(f"Das Dokument enthält mehrere Helden; bitte hero_name auswählen: {names}")
    return heroes[0]


def _visual_data(hero):
    basis = _child(hero, "basis")
    species = _child(basis, "rasse")
    appearance = _child(species, "aussehen")
    if appearance is None:
        appearance = _child(basis, "aussehen")
    size = _child(species, "groesse")
    if size is None:
        size = _child(basis, "groesse")
    visual = {
        key: _attribute(appearance, source)
        for key, source in (("age", "alter"), ("eye_color", "augenfarbe"),
                            ("hair_color", "haarfarbe"), ("title", "titel"))
    }
    visual.update(height_cm=_attribute(size, "value"), weight_kg=_attribute(size, "gewicht"))
    descriptions = []
    if appearance is not None:
        keys = sorted((key for key in appearance.attrib if re.fullmatch(r"aussehentext\d+", key)),
                      key=lambda key: int(key.removeprefix("aussehentext")))
        descriptions = [_attribute(appearance, key) for key in keys]
        descriptions.extend(_clean(child.text) for child in appearance if _tag(child) == "aussehentext")
    professions = _unique(_label(item) for item in _children(_child(basis, "ausbildungen"), "ausbildung"))
    traits = []
    for item in _children(_child(hero, "vt"), "vorteil"):
        name, value = _attribute(item, "name"), _attribute(item, "value")
        if not any(term in f"{name} {value}".casefold() for term in _VISIBLE_TRAITS):
            continue
        # Mechanical levels and damage dice do not describe the visible character.
        value = re.sub(r"\([^()]*\d+\s*[wWdD]\s*\d+[^()]*(?:\([^()]*\)[^()]*)*\)", "", value).strip()
        if re.fullmatch(r"[+-]?\d+(?:[.,]\d+)?", value):
            value = ""
        traits.append(f"{name}: {value}" if value else name)
    equipment = [_attribute(item, "name") for item in _children(_child(hero, "gegenstände"), "gegenstand")
                 if _attribute(item, "anzahl") != "0"]
    equipment.extend(_attribute(item, "waffenname")
                     for item in _children(_child(hero, "ausrüstungen"), "heldenausruestung"))
    result = {
        "source_format": "helden_xml", "name": _attribute(hero, "name"),
        "gender": _attribute(_child(basis, "geschlecht"), "name"),
        "species": _label(species), "culture": _label(_child(basis, "kultur")),
        "professions": professions, "appearance": {key: value for key, value in visual.items() if value},
        "appearance_text": _unique(descriptions), "visible_traits": _unique(traits),
        "equipment": _unique(equipment),
    }
    return {key: value for key, value in result.items() if value}


def load_hero_document(document_path="", document_text="", hero_name=""):
    """Extract reference data; no document text is treated as executable instructions.

    Nonblank ``document_text`` takes precedence over a file. XML is parsed in its
    declared encoding; plain-text files accept UTF-8, UTF-16/32 BOMs and CP1252.
    Inventory entries describe available equipment, not necessarily worn items.
    """
    text = str(document_text or "").strip().lstrip("\ufeff").strip()
    if text:
        if len(text.encode("utf-8")) > MAX_DOCUMENT_BYTES:
            raise ValueError("Der Heldentext ist größer als 4 MiB.")
        if re.match(r"^(?:[A-Za-z]:[\\/]|\\\\|/|~/)", text) and re.search(r"\.(xml|txt)[\"']?$", text, re.I) and "\n" not in text:
            raise ValueError("Bitte den Dateipfad in document_path eintragen; document_text erwartet den Dateiinhalt.")
        data, suffix = text, ".xml" if text.startswith("<") else ".txt"
    else:
        data, suffix = _read_file(document_path)
        if suffix == ".txt":
            try:
                data = _decode_text(data).strip()
            except UnicodeError as error:
                raise ValueError("Die Textdatei hat keine unterstützte Textkodierung.") from error
    if suffix == ".xml" or (isinstance(data, str) and data.lstrip().startswith("<")):
        return _visual_data(_xml_hero(data, hero_name))
    text = str(data).strip()
    if not text:
        raise ValueError("Das Heldendokument ist leer.")
    return {"source_format": "text", "description": text}


def hero_filename_name(hero, document_path="", document_text=""):
    """Name the parsed hero, independently of Ollama and the frontend's next selection."""
    name = str(hero.get("name") or "").strip()
    inline_text = str(document_text or "").strip().lstrip("\ufeff").strip()
    if not name and not inline_text:
        name = Path(hero_path_text(document_path)).stem
    # Percent signs must not reach SaveImage's %width% / %batch_num% expansion.
    name = unicodedata.normalize("NFC", name)
    name = re.sub(r'[<>:"/\\|?*%\x00-\x1f\x7f-\x9f\s]+', "_", name)
    name = re.sub(r"_+", "_", name).strip(" ._")[:100].rstrip(" ._")
    if not name:
        return "Unbenannter_Held"
    if re.match(r"^(CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\.|$)", name, re.I):
        name = "_" + name
    return name[:100].rstrip(" ._")


def hero_document_fingerprint(document_path="", document_text="", hero_name="", **kwargs):
    """Invalidate ComfyUI's node cache when file contents change, including in-place edits."""
    digest = hashlib.sha256()
    digest.update(str(hero_name or "").encode("utf-8"))
    text = str(document_text or "").strip().lstrip("\ufeff").strip()
    if text:
        digest.update(b"\0text\0" + text.encode("utf-8"))
    else:
        try:
            data, suffix = _read_file(document_path)
            digest.update(b"\0file\0" + suffix.encode("ascii") + b"\0" + data)
        except (ValueError, OSError) as error:
            digest.update(b"\0error\0" + str(error).encode("utf-8"))
    return digest.hexdigest()
