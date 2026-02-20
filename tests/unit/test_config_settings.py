"""Unit tests for environment settings parsing."""

from core.config import Settings


def test_allowed_origins_from_json_env(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", '["http://localhost:3000", "https://example.com"]')
    settings = Settings()
    assert settings.ALLOWED_ORIGINS == ["http://localhost:3000", "https://example.com"]


def test_allowed_origins_from_csv_env(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:3000, https://example.com")
    settings = Settings()
    assert settings.ALLOWED_ORIGINS == ["http://localhost:3000", "https://example.com"]


def test_allowed_origins_from_empty_env(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "")
    settings = Settings()
    assert settings.ALLOWED_ORIGINS == []
