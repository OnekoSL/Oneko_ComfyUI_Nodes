"""Read-only directory listing for the hero portrait's original-file picker."""

import asyncio
from ipaddress import IPv6Address, ip_address
import os
from pathlib import Path
import re
import stat

from .hero_document import (
    HERO_EXTENSIONS,
    HeroPathOutsideRoots,
    hero_document_roots,
    hero_path_text,
    in_resolved_hero_roots,
    parse_hero_path,
    resolve_hero_path,
)

try:
    from aiohttp import web
    from server import PromptServer
except ImportError:
    web = None
    PromptServer = None


class HeroBrowseError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def _filesystem_error(error):
    if isinstance(error, PermissionError):
        return HeroBrowseError("Dieser Ordner kann nicht gelesen werden: Zugriff verweigert.", 403)
    if isinstance(error, (FileNotFoundError, NotADirectoryError)):
        return HeroBrowseError("Der angegebene Ordner oder die Heldendatei wurde nicht gefunden.", 404)
    return HeroBrowseError("Der angegebene Pfad kann nicht gelesen werden. Bitte einen verfügbaren Ordner auswählen.")


def _sort_name(entry):
    return tuple(int(part) if part.isdigit() else part.casefold()
                 for part in re.split(r"(\d+)", entry["name"]))


def list_hero_files(path="", input_directory=None):
    """List approved folders and XML/TXT files without opening the files."""
    roots = hero_document_roots(input_directory)
    if not roots:
        raise HeroBrowseError("Für Heldendokumente ist kein freigegebener Ordner verfügbar.", 404)
    value = hero_path_text(path)
    try:
        candidate = parse_hero_path(value) if value else next(iter(roots.values()))
        if not candidate.is_absolute():
            raise HeroBrowseError("Bitte einen vollständigen Pfad zu einem freigegebenen Ordner angeben.")
        candidate = resolve_hero_path(candidate, roots)
        mode = candidate.stat().st_mode
        selected = None
        if stat.S_ISREG(mode):
            if candidate.suffix.casefold() not in HERO_EXTENSIONS:
                raise HeroBrowseError("Bitte einen Ordner oder eine .xml- bzw. .txt-Heldendatei auswählen.")
            selected = str(candidate)
            directory = candidate.parent
        elif stat.S_ISDIR(mode):
            directory = candidate
        else:
            raise HeroBrowseError("Dieser Pfad ist kein Ordner und keine reguläre Heldendatei.")

        directories, files = [], []
        with os.scandir(directory) as entries:
            for entry in entries:
                try:
                    child = resolve_hero_path(Path(entry.path), roots)
                    entry_stat = child.stat()
                    if stat.S_ISDIR(entry_stat.st_mode):
                        directories.append({"name": entry.name, "path": entry.path})
                    elif (stat.S_ISREG(entry_stat.st_mode) and Path(entry.name).suffix.casefold() in HERO_EXTENSIONS
                          and child.suffix.casefold() in HERO_EXTENSIONS):
                        files.append({"name": entry.name, "path": entry.path, "size": entry_stat.st_size})
                except (OSError, RuntimeError, ValueError):
                    # Unavailable or excluded children do not hide readable siblings.
                    continue
    except HeroBrowseError:
        raise
    except HeroPathOutsideRoots as error:
        raise HeroBrowseError(str(error), 403) from error
    except (OSError, RuntimeError) as error:
        raise _filesystem_error(error) from error
    except ValueError as error:
        raise HeroBrowseError(str(error)) from error

    parent = directory.parent
    approved = dict.fromkeys(roots.values())
    return {
        "directory": str(directory),
        "parent": str(parent) if parent != directory and in_resolved_hero_roots(parent, approved) else None,
        "roots": [{"name": str(root), "path": str(root)} for root in approved],
        "directories": sorted(directories, key=_sort_name),
        "files": sorted(files, key=_sort_name),
        "selected": selected,
    }


def _is_local_request(request):
    # Use the socket peer so request headers cannot change the access decision.
    transport = request.transport
    peer = transport.get_extra_info("peername") if transport is not None else None
    if not isinstance(peer, tuple) or not peer or not isinstance(peer[0], str):
        return False
    try:
        address = ip_address(peer[0])
    except ValueError:
        return False
    if isinstance(address, IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_loopback


async def get_hero_files(request):
    if not _is_local_request(request):
        return web.json_response(
            {"error": "Die Dateiauswahl ist nur über eine lokale Verbindung verfügbar."},
            status=403,
            headers={"Cache-Control": "no-store"},
        )
    try:
        result = await asyncio.to_thread(list_hero_files, request.query.get("path", ""))
    except HeroBrowseError as error:
        return web.json_response({"error": str(error)}, status=error.status)
    return web.json_response(result, headers={"Cache-Control": "no-store"})


def _register_routes():
    if web is None or PromptServer is None:
        return
    if not hasattr(PromptServer, "instance") or PromptServer.instance is None:
        return
    PromptServer.instance.routes.get("/oneko/hero/files")(get_hero_files)


_register_routes()
