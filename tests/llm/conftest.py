from types import SimpleNamespace

import pytest

from ai_lab.config import get_settings
from ai_lab.llm import gemini


@pytest.fixture(autouse=True)
def fake_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def install_models(monkeypatch):
    """Install a fake `models` object as the Gemini client: install_models(fake).

    Patches where the name is looked up (gemini._client), not where it is
    re-exported (ai_lab.llm), which would have no effect.
    """

    def install(models):
        monkeypatch.setattr(gemini, "_client", lambda: SimpleNamespace(models=models))

    return install
