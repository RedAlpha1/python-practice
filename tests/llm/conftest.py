import logging
import time
from types import SimpleNamespace

import pytest

from ai_lab.config import get_settings
from ai_lab.llm import gemini
from ai_lab.llm.usage import usage_logger


@pytest.fixture(autouse=True)
def fake_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """The public calls retry transient errors; never really wait in tests."""
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)


@pytest.fixture(autouse=True)
def detach_usage_handlers():
    """A test that calls log_usage_to must not leave a file handler for the next one."""
    yield
    for handler in list(usage_logger.handlers):
        if not isinstance(handler, logging.NullHandler):
            usage_logger.removeHandler(handler)
            handler.close()


@pytest.fixture
def install_models(monkeypatch):
    """Install a fake `models` object as the Gemini client: install_models(fake).

    Patches where the name is looked up (gemini._client), not where it is
    re-exported (ai_lab.llm), which would have no effect.
    """

    def install(models):
        monkeypatch.setattr(gemini, "_client", lambda: SimpleNamespace(models=models))

    return install
