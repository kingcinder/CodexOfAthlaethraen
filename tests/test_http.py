"""HTTP integration tests for the Genie Lamp service."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from contextlib import contextmanager
from typing import Generator, Optional

import pytest
import requests

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7861
DEFAULT_BASE_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"


@contextmanager
def _start_server(base_url: str) -> Generator[Optional[subprocess.Popen[str]], None, None]:
    if os.environ.get("BASE_URL"):
        # Assume an external server is already running.
        yield None
        return

    host = DEFAULT_HOST
    port = str(DEFAULT_PORT)
    env = os.environ.copy()
    env.setdefault("GENIE_LAMP_HOST", host)
    env.setdefault("GENIE_LAMP_PORT", port)
    env.setdefault("ALLOW_NETWORK", env.get("ALLOW_NETWORK", "0"))

    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "genie_lamp.main:app",
        "--host",
        host,
        "--port",
        port,
    ]
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_health(base_url)
        yield process
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def _wait_for_health(base_url: str, retries: int = 30, delay: float = 0.5) -> None:
    for _ in range(retries):
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code == 200:
                return
        except requests.RequestException:
            time.sleep(delay)
            continue
        time.sleep(delay)
    raise RuntimeError("Service health check did not succeed within the timeout window")


def _determine_base_url() -> str:
    override = os.environ.get("BASE_URL")
    if override:
        return override.rstrip("/")
    return DEFAULT_BASE_URL


@pytest.mark.integration
def test_service_endpoints():
    base_url = _determine_base_url()

    with _start_server(base_url):
        health_resp = requests.get(f"{base_url}/health", timeout=5)
        assert health_resp.status_code == 200
        payload = health_resp.json()
        assert payload.get("ok") is True

        version_resp = requests.get(f"{base_url}/version", timeout=5)
        assert version_resp.status_code == 200
        version_json = version_resp.json()
        assert version_json.get("name") == "genie-lamp"
        assert isinstance(version_json.get("version"), str)

        diag_resp = requests.get(f"{base_url}/diag", timeout=5)
        assert diag_resp.status_code == 200
        diag_json = diag_resp.json()
        for key in ("python", "platform", "retrieval", "memory"):
            assert key in diag_json

        ready_resp = requests.get(f"{base_url}/ready", timeout=10)
        if ready_resp.status_code != 200:
            detail = ready_resp.json().get("detail") if ready_resp.headers.get("content-type", "").startswith("application/json") else ready_resp.text
            pytest.fail(
                "Expected /ready to return 200. "
                "Ensure sentence-transformer artifacts are present (run scripts/fetch-model.ps1 -AllowNetwork) "
                "or set ALLOW_NETWORK=1 before starting the service. "
                f"Response detail: {detail}"
            )
        ready_json = ready_resp.json()
        assert ready_json.get("status") == "ready"
