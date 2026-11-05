"""Settings for every test."""

import hashlib

import pytest

import app.settings as settings

REAL_ENV = settings.ENV_FILE


def _fingerprint():
    return hashlib.sha256(REAL_ENV.read_bytes()).hexdigest() if REAL_ENV.exists() else None


@pytest.fixture(autouse=True)
def never_touch_the_real_env(tmp_path, monkeypatch):
    """No test may change your real .env (it holds your keys): saving goes to a temporary file,
    and the test fails if the real file changed anyway."""
    before = _fingerprint()
    monkeypatch.setattr(settings, "ENV_FILE", tmp_path / "test.env")
    yield
    assert _fingerprint() == before, "a test changed the real .env file"
