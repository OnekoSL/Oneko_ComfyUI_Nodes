import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from Oneko_ComfyUI_Nodes.nodes import hero_file_browser as browser


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
    assert result["parent"] == str(tmp_path.parent)


def test_empty_path_uses_comfy_input_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(browser.folder_paths, "get_input_directory", lambda: str(tmp_path))
    assert browser.list_hero_files()["directory"] == str(tmp_path)
    assert browser.list_hero_files("  ", input_directory=tmp_path)["directory"] == str(tmp_path)


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
    class Unreadable:
        def is_dir(self):
            raise PermissionError("denied")

    class Readable:
        name = "Held.xml"
        path = str(tmp_path / name)
        def is_dir(self):
            return False
        def is_file(self):
            return True
        def stat(self):
            return SimpleNamespace(st_size=12)

    class Entries:
        def __enter__(self):
            return iter([Unreadable(), Readable()])
        def __exit__(self, *args):
            pass

    monkeypatch.setattr(browser.os, "scandir", lambda _: Entries())
    result = browser.list_hero_files(tmp_path)
    assert result["files"] == [{"name": "Held.xml", "path": str(tmp_path / "Held.xml"), "size": 12}]


def test_root_has_no_parent_and_roots_are_absolute(tmp_path):
    result = browser.list_hero_files(Path(tmp_path.anchor))
    assert result["parent"] is None
    assert all(Path(item["path"]).is_absolute() for item in result["roots"])
    assert any(item["path"] == tmp_path.anchor for item in result["roots"])


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
