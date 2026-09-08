from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def offline_ollama():
    with patch("urllib.request.urlopen", side_effect=OSError("Network disabled during tests")):
        yield
