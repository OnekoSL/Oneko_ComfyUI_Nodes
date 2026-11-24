import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import folder_paths

from Oneko_ComfyUI_Nodes.nodes import hero_file_browser as browser
from Oneko_ComfyUI_Nodes.nodes import hero_document as document


@pytest.fixture(autouse=True)
def core_input_root(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path))
    monkeypatch.setattr(folder_paths, "base_path", str(tmp_path))


def test_lists_only_hero_files_and_folders_in_natural_order_without_modifying_files(tmp_path):
    (tmp_path / "Ordner 10").mkdir()
    (tmp_path / "Ordner 2").mkdir()
    sources = {"10 Zauberer.XML": b"not parsed XML", "2 Elfin.txt": "Ältere Heldin".encode("utf-8"),
               "Bild.png": b"image", "skript.py": b"do not execute"}
    for name, content in sources.items():
        (tmp_path / name).write_bytes(content)
    result = browser.list_hero_files(str(tmp_path))
    assert result["directory"] == str(tmp_path.resolve())
    assert result["selected"] is None
    assert [item["name"] for item in result["directories"]] == ["Ordner 2", "Ordner 10"]
    assert [item["name"] for item in result["files"]] == ["2 Elfin.txt", "10 Zauberer.XML"]
    assert result["files"][0]["size"] == len(sources["2 Elfin.txt"])
    assert all(Path(item["path"]).is_absolute() for item in result["directories"] + result["files"])
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()} == sources


def test_existing_document_opens_its_parent_and_preserves_selection(tmp_path):
    source = tmp_path / "Held mit Leerzeichen.XML"
    source.write_text("<held/>", encoding="utf-8")
    result = browser.list_hero_files(f'"{source}"')
    assert result["directory"] == str(tmp_path)
    assert result["selected"] == str(source)
    assert result["parent"] is None


def test_empty_path_uses_comfy_input_directory(tmp_path):
    assert browser.list_hero_files()["directory"] == str(tmp_path)
    assert browser.list_hero_files("  ")["directory"] == str(tmp_path)


@pytest.mark.parametrize("path", ["relative/path", "https://example.com/hero.xml", "file:///J:/hero.xml", "bad\0path"])
def test_invalid_and_remote_paths_are_rejected(path):
    with pytest.raises(browser.HeroBrowseError) as error:
        browser.list_hero_files(path)
    assert error.value.status == 400


def test_missing_path_and_non_document_file_have_actionable_errors(tmp_path):
    with pytest.raises(browser.HeroBrowseError) as error:
        browser.list_hero_files(tmp_path / "missing")
    assert error.value.status == 404
    source = tmp_path / "other.pdf"
    source.write_bytes(b"pdf")
    with pytest.raises(browser.HeroBrowseError, match=".xml") as error:
        browser.list_hero_files(source)
    assert error.value.status == 400


def test_unreadable_directory_reports_access_denied(tmp_path, monkeypatch):
    def denied(_):
        raise PermissionError("denied")
    monkeypatch.setattr(browser.os, "scandir", denied)
    with pytest.raises(browser.HeroBrowseError, match="Zugriff verweigert") as error:
        browser.list_hero_files(tmp_path)
    assert error.value.status == 403


def test_inaccessible_child_does_not_hide_other_files(tmp_path, monkeypatch):
    (tmp_path / "Held.xml").write_bytes(b"<held></held>")
    unavailable = tmp_path / "Unavailable.xml"
    unavailable.write_bytes(b"<held/>")
    original_stat = Path.stat

    def stat_with_inaccessible_child(path, *args, **kwargs):
        if path == unavailable:
            raise PermissionError("denied")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat_with_inaccessible_child)
    result = browser.list_hero_files(tmp_path)
    assert result["files"] == [{"name": "Held.xml", "path": str(tmp_path / "Held.xml"), "size": 13}]


def test_root_has_no_parent_and_roots_include_only_allowed_directories(tmp_path, monkeypatch):
    core_input = tmp_path / "input"
    library = tmp_path / "hero library"
    core_input.mkdir()
    library.mkdir()
    (library / "adventurers").mkdir()
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(core_input))
    (tmp_path / "oneko_hero_document_paths.json").write_text(
        json.dumps({"roots": [str(library)]}), encoding="utf-8",
    )
    result = browser.list_hero_files(library)
    assert result["parent"] is None
    assert {Path(item["path"]) for item in result["roots"]} == {core_input.resolve(), library.resolve()}
    assert browser.list_hero_files(library / "adventurers")["parent"] == str(library.resolve())


def test_picker_rejects_parent_traversal_and_similarly_named_sibling(tmp_path, monkeypatch):
    root = tmp_path / "heroes"
    sibling = tmp_path / "heroes_other"
    root.mkdir()
    sibling.mkdir()
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(root))
    for candidate in (sibling, root / ".." / sibling.name):
        with pytest.raises(browser.HeroBrowseError, match="freigegebenen Ordner") as error:
            browser.list_hero_files(candidate)
        assert error.value.status == 403


def test_picker_hides_links_to_files_and_directories_outside_roots(tmp_path, monkeypatch):
    root = tmp_path / "heroes"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "other.xml").write_text("<held/>", encoding="utf-8")
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(root))
    links = root / "other.xml", root / "outside directory"
    try:
        links[0].symlink_to(outside / "other.xml")
        links[1].symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"Symbolic links unavailable: {error}")
    listing = browser.list_hero_files(root)
    assert not listing["files"]
    assert not listing["directories"]
    for link in links:
        with pytest.raises(browser.HeroBrowseError, match="freigegebenen Ordner") as error:
            browser.list_hero_files(link)
        assert error.value.status == 403


@pytest.mark.skipif(os.name != "nt", reason="Windows directory junctions")
def test_windows_junction_cannot_escape_the_input_root(tmp_path, monkeypatch):
    winapi = pytest.importorskip("_winapi")
    root = tmp_path / "heroes"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "other.xml").write_text("<held/>", encoding="utf-8")
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(root))
    junction = root / "junction"
    try:
        winapi.CreateJunction(str(outside), str(junction))
    except (AttributeError, OSError) as error:
        pytest.skip(f"Windows junctions unavailable: {error}")

    assert not browser.list_hero_files(root)["directories"]
    with pytest.raises(browser.HeroBrowseError, match="freigegebenen Ordner") as error:
        browser.list_hero_files(junction)
    assert error.value.status == 403
    with pytest.raises(ValueError, match="freigegebenen Ordner"):
        document.load_hero_document(document_path=junction / "other.xml")


@pytest.mark.parametrize("peer", [("127.0.0.1", 1234), ("::1", 1234, 0, 0), ("::ffff:127.0.0.1", 1234, 0, 0)])
def test_route_returns_listing_and_clear_json_errors_for_loopback_peers(tmp_path, peer):
    request = SimpleNamespace(
        query={"path": str(tmp_path)},
        transport=SimpleNamespace(get_extra_info={"peername": peer}.get),
    )
    result = asyncio.run(browser.get_hero_files(request))
    assert result.status == 200
    assert result.headers["Cache-Control"] == "no-store"
    assert json.loads(result.body)["directory"] == str(tmp_path)
    request.query["path"] = str(tmp_path / "missing")
    result = asyncio.run(browser.get_hero_files(request))
    assert result.status == 404
    assert "nicht gefunden" in json.loads(result.body)["error"]


def test_route_rejects_paths_outside_the_roots_even_on_loopback(tmp_path, monkeypatch):
    root = tmp_path / "input"
    root.mkdir()
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(root))

    def reject_scan(path):
        pytest.fail(f"The route attempted to list a disallowed directory: {path}")

    monkeypatch.setattr(browser.os, "scandir", reject_scan)
    request = SimpleNamespace(
        query={"path": str(tmp_path)},
        transport=SimpleNamespace(get_extra_info={"peername": ("127.0.0.1", 1234)}.get),
    )
    result = asyncio.run(browser.get_hero_files(request))
    assert result.status == 403
    assert "freigegebenen Ordner" in json.loads(result.body)["error"]


@pytest.mark.parametrize(
    "peer",
    [("192.0.2.1", 1234), ("2001:db8::1", 1234, 0, 0), ("::ffff:192.0.2.1", 1234, 0, 0),
     ("localhost", 1234), None, (), "/tmp/local.socket"],
)
def test_route_rejects_other_peers_before_accessing_path(peer):
    request = SimpleNamespace(
        remote="127.0.0.1",
        headers={"Forwarded": "for=127.0.0.1", "X-Forwarded-For": "127.0.0.1"},
        transport=SimpleNamespace(get_extra_info={"peername": peer}.get),
    )
    result = asyncio.run(browser.get_hero_files(request))
    assert result.status == 403
    assert result.headers["Cache-Control"] == "no-store"
    assert "lokale Verbindung" in json.loads(result.body)["error"]


def test_route_rejects_requests_without_transport_before_accessing_path():
    request = SimpleNamespace(transport=None)
    result = asyncio.run(browser.get_hero_files(request))
    assert result.status == 403


def test_routes_register_only_with_available_server(monkeypatch):
    registered = []
    routes = SimpleNamespace(get=lambda path: lambda handler: registered.append((path, handler)))
    monkeypatch.setattr(browser, "PromptServer", SimpleNamespace(instance=SimpleNamespace(routes=routes)))
    browser._register_routes()
    assert registered == [("/oneko/hero/files", browser.get_hero_files)]
    monkeypatch.setattr(browser, "PromptServer", SimpleNamespace())
    browser._register_routes()
    assert len(registered) == 1
