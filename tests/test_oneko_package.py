import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
from PIL import Image
import torch

import Oneko_ComfyUI_Nodes as package
from Oneko_ComfyUI_Nodes.nodes import load_image_with_subfolders as loader
from Oneko_ComfyUI_Nodes.nodes import noise_sampler_core as noise
from Oneko_ComfyUI_Nodes.tools.migrate_workflow import migrate_workflow


def test_package_contracts_and_frontend_targets():
    assert len(package.NODE_CLASS_MAPPINGS) == 43
    assert package.NODE_CLASS_MAPPINGS.keys() == package.NODE_DISPLAY_NAME_MAPPINGS.keys()
    for name, cls in package.NODE_CLASS_MAPPINGS.items():
        assert name.startswith("Oneko")
        assert cls.CATEGORY.startswith("Oneko/")
        assert callable(getattr(cls, cls.FUNCTION))
    repo = Path(package.__file__).parent
    mapping = json.loads((repo / "migration/node_id_map.json").read_text(encoding="utf-8"))
    assert set(package.NODE_CLASS_MAPPINGS) - set(mapping.values()) == {
        "OnekoAudioTrackSettings", "OnekoAudioTimelineMixer", "OnekoOllamaVerseMaker",
    }
    frontend = "\n".join(f.read_text(encoding="utf-8") for f in (repo / "web").iterdir())
    assert "/oneko/ollama/models" in frontend
    assert "Nukun" not in frontend
    assert "/nukun/" not in frontend


@pytest.mark.parametrize("profile", ["invalid", "pony_v7_unknown", "illustrious_unknown"])
def test_invalid_noise_profile_is_not_silently_gaussian(profile):
    with pytest.raises(ValueError, match="Unknown Oneko noise profile"):
        noise.make_noise_generator(1, "cpu", profile)


def test_basic_noise_profile_keeps_the_existing_generator_result():
    latent = {"samples": torch.zeros(1, 4, 8, 8)}
    for profile in noise.NOISE_TYPES:
        actual = noise.make_noise_generator(123, "cpu", profile).generate_noise(latent)
        expected = noise.OnekoRandomNoise(123, "cpu", profile).generate_noise(latent)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_recursive_image_menu_omits_text_and_clipspace(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "clipspace").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "nested/image.png")
    Image.new("RGB", (8, 8)).save(tmp_path / "clipspace/temporary.png")
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")
    with patch.object(loader.folder_paths, "get_input_directory", return_value=str(tmp_path)):
        files = loader.OnekoLoadImageWithSubfolders.INPUT_TYPES()["required"]["image"][0]
    assert files == ["nested/image.png"]


def test_recursive_loader_preserves_animated_image_batch(tmp_path):
    path = tmp_path / "two_frames.gif"
    Image.new("RGB", (16, 16), "red").save(
        path, save_all=True, append_images=[Image.new("RGB", (16, 16), "blue")], duration=100, loop=0
    )
    with patch.object(loader.folder_paths, "get_annotated_filepath", return_value=str(path)):
        images, masks = loader.OnekoLoadImageWithSubfolders().load_image("two_frames.gif")
    assert images.shape == (2, 16, 16, 3)
    assert masks.shape[0] == 2
    assert images[0, 0, 0, 0] > images[0, 0, 0, 2]
    assert images[1, 0, 0, 2] > images[1, 0, 0, 0]


def test_migration_handles_subgraphs_without_rewriting_prompt_text():
    original = {
        "nodes": [{"id": 1, "type": "NukunOllamaPromptRefiner", "widgets_values": ["Nukun is a character"]}],
        "definitions": {"subgraphs": [{"nodes": [{"id": 2, "type": "NukunWan22SegmentStore"}]}]},
    }
    migrated, counts, unsupported = migrate_workflow(original)
    assert migrated["nodes"][0]["type"] == "OnekoOllamaPromptRefiner"
    assert migrated["nodes"][0]["widgets_values"] == ["Nukun is a character"]
    assert migrated["definitions"]["subgraphs"][0]["nodes"][0]["type"] == "OnekoWan22SegmentStore"
    assert original["nodes"][0]["type"] == "NukunOllamaPromptRefiner"
    assert sum(counts.values()) == 2
    assert not unsupported


def test_migration_reports_excluded_nodes_and_preserves_external_nodes():
    original = {"1": {"class_type": "NukunAdvancedNoiseSampler"}, "2": {"class_type": "SamplerSPEED"}}
    migrated, counts, unsupported = migrate_workflow(original)
    assert migrated == original
    assert not counts
    assert unsupported == ["NukunAdvancedNoiseSampler"]


def test_migration_cli_dry_run_and_exclusive_output(tmp_path):
    source = tmp_path / "old.json"
    source.write_text(json.dumps({"1": {"class_type": "NukunOllamaPromptRefiner"}}), encoding="utf-8")
    original = source.read_bytes()
    tool = Path(package.__file__).parent / "tools/migrate_workflow.py"
    command = [sys.executable, str(tool), str(source)]
    assert subprocess.run(command, capture_output=True).returncode == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["old.json"]
    target = tmp_path / "oneko.json"
    assert subprocess.run([*command, "--output", str(target)], capture_output=True).returncode == 0
    assert json.loads(target.read_text(encoding="utf-8"))["1"]["class_type"] == "OnekoOllamaPromptRefiner"
    before = target.read_bytes()
    assert subprocess.run([*command, "--output", str(target)], capture_output=True).returncode != 0
    assert target.read_bytes() == before
    assert source.read_bytes() == original


def test_migration_cli_does_not_write_a_partial_workflow(tmp_path):
    source = tmp_path / "unsupported.json"
    source.write_text(json.dumps({"1": {"class_type": "NukunAdvancedNoiseSampler"}}), encoding="utf-8")
    target = tmp_path / "oneko.json"
    tool = Path(package.__file__).parent / "tools/migrate_workflow.py"
    result = subprocess.run([sys.executable, str(tool), str(source), "--output", str(target)], capture_output=True)
    assert result.returncode == 2
    assert not target.exists()


def test_examples_have_registered_oneko_nodes_and_valid_links():
    for path in (Path(package.__file__).parent / "examples").rglob("*.json"):
        workflow = json.loads(path.read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in workflow["nodes"]}
        for node in nodes.values():
            assert not node["type"].startswith("Nukun"), path.name
            if node["type"].startswith("Oneko"):
                assert node["type"] in package.NODE_CLASS_MAPPINGS, path.name
        for link_id, source, output, target, input_slot, _ in workflow["links"]:
            assert nodes[target]["inputs"][input_slot]["link"] == link_id, path.name
            assert link_id in nodes[source]["outputs"][output]["links"], path.name
