import os
from pathlib import Path


ONEKO_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(os.environ["ONEKO_COMFY_ROOT"])
