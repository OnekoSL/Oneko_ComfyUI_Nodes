"""Read-only directory listing for the hero portrait's original-file picker."""

import asyncio
from ipaddress import IPv6Address, ip_address
import os
from pathlib import Path
import re
import stat

import folder_paths

try:
    from aiohttp import web
    from server import PromptServer
except ImportError:
    web = None
    PromptServer = None


HERO_EXTENSIONS = frozenset((".xml", ".txt"))


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


def _roots(directory):
    """List drive names without probing removable or disconnected drive contents."""
    paths = []
    if os.name == "nt":
        import ctypes
        try:
            mask = ctypes.windll.kernel32.GetLogicalDrives()
            paths = [f"{chr(65 + index)}:\\" for index in range(26) if mask & (1 << index)]
        except (AttributeError, OSError):
            pass
    else:
        paths = [str(Path("/")), str(Path.home())]
    # An explicitly browsed UNC share may not appear in Windows' drive mask.
    if directory.anchor and directory.anchor not in paths:
        paths.append(directory.anchor)
    return [{"name": value, "path": value} for value in dict.fromkeys(paths)]


def list_hero_files(path="", input_directory=None):
    """List folders and XML/TXT files without opening, copying or modifying them."""
    value = str(path or "").strip()
    if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    if not value:
        value = str(input_directory if input_directory is not None else folder_paths.get_input_directory())
    if "\0" in value or re.match(r"^[a-z][a-z0-9+.-]*://", value, re.I):
        raise HeroBrowseError("Bitte einen lokalen Ordnerpfad statt einer URL angeben.")
    try:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            raise HeroBrowseError("Bitte einen vollständigen Ordnerpfad angeben, zum Beispiel J:\\Helden.")
        candidate = candidate.resolve(strict=True)
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
                    if entry.is_dir():
                        directories.append({"name": entry.name, "path": entry.path})
                    elif Path(entry.name).suffix.casefold() in HERO_EXTENSIONS and entry.is_file():
                        files.append({"name": entry.name, "path": entry.path, "size": entry.stat().st_size})
                except OSError:
                    # A vanished or inaccessible child must not hide its readable siblings.
                    continue
    except HeroBrowseError:
        raise
    except (OSError, RuntimeError) as error:
        raise _filesystem_error(error) from error
    except ValueError as error:
        raise HeroBrowseError("Der Ordnerpfad ist ungültig.") from error

    return {
        "directory": str(directory),
        "parent": str(directory.parent) if directory.parent != directory else None,
        "roots": _roots(directory),
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
