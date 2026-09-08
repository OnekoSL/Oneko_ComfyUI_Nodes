"""Run the Oneko test suite against an existing ComfyUI checkout on CPU."""

import argparse
import os
from pathlib import Path
import sys


def main():
    repo = Path(__file__).resolve().parents[1]
    default = repo.parent.parent if repo.parent.name == "custom_nodes" else repo.parent / "ComfyUI"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, default=Path(os.environ.get("ONEKO_COMFY_ROOT", default)))
    options, pytest_args = parser.parse_known_args()
    comfy_root = options.comfy_root.resolve()
    if not (comfy_root / "folder_paths.py").is_file():
        parser.error("Pass --comfy-root pointing to an existing ComfyUI checkout.")
    os.environ["ONEKO_COMFY_ROOT"] = str(comfy_root)
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    for path in (repo.parent, repo, comfy_root):
        sys.path.insert(0, str(path))

    # Choose CPU before importing model_management or any node modules.
    from comfy.cli_args import args
    args.cpu = True
    # Match ComfyUI startup: the core `nodes` module exists before custom nodes.
    import nodes
    sys.path.insert(0, str(repo))
    import pytest
    return pytest.main([str(repo / "tests"), *pytest_args])


if __name__ == "__main__":
    raise SystemExit(main())
