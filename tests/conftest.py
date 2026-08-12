"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture(scope="session")
def api_key():
    return "voyage-dev-key-2024"


@pytest.fixture(scope="session")
def headers(api_key):
    return {"X-API-Key": api_key}
