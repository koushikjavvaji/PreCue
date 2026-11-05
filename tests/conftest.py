"""Settings for every test."""

import hashlib
import os
import subprocess
import sys
import time

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


@pytest.fixture(scope="session", autouse=True)
def quiet_screen_corner():
    """For the real-window UI tests: a grey panel where the pop-up appears, so whatever is on
    your screen there cannot be mistaken for an alert (see tests/backdrop_app.py)."""
    if sys.platform != "darwin" or os.getenv("PRECUE_UI_TESTS") != "1":
        yield
        return
    process = subprocess.Popen([sys.executable, "-m", "tests.backdrop_app"], stdout=subprocess.PIPE, text=True)
    process.stdout.readline()
    time.sleep(0.5)  # let the window server draw it
    yield
    process.terminate()
    process.wait(timeout=5)
