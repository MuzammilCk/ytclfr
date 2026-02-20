"""
tests/integration/test_analytics.py

Integration tests for the analytics dashboard endpoints.
Run with: pytest tests/ -v

These tests verify that the analytics routes are registered (bug #1 fixed),
return the correct schema shape, and handle missing data gracefully.

The conftest.py in this directory stubs out celery/gssapi and heavy ML deps
before any app module is loaded, so no system-level Kerberos is needed.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import AsyncClient, ASGITransport

# Module-level app import — stubs are already in sys.modules via conftest.py
from main import app  # noqa: E402
from db.database import get_db_session  # noqa: E402


def _make_mock_db():
    """Return a mock AsyncSession that returns empty result-sets."""
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar.return_value = 0
    mock_result.scalar_one_or_none.return_value = None
    mock_result.mappings.return_value.all.return_value = []
    mock_result.__iter__ = MagicMock(return_value=iter([]))
    mock_session.execute = AsyncMock(return_value=mock_result)
    return mock_session


@pytest.fixture
async def client():
    """HTTP test client with the DB dependency overridden to avoid real Postgres."""
    mock_session = _make_mock_db()

    async def override_get_db():
        yield mock_session

    app.dependency_overrides[get_db_session] = override_get_db
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as ac:
            yield ac
    finally:
        app.dependency_overrides.pop(get_db_session, None)


class TestAnalyticsSummary:
    async def test_summary_endpoint_registered(self, client):
        """Verify the route is registered — must NOT return 404 (fixed bug)."""
        resp = await client.get("/api/v1/analytics/summary")
        assert resp.status_code != 404, "Analytics router was not registered in main.py"

    async def test_summary_returns_valid_schema(self, client):
        """When DB is mocked, response should include expected top-level keys."""
        resp = await client.get("/api/v1/analytics/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        assert "total_analyses" in data
        assert "completed_analyses" in data

    async def test_summary_accepts_days_filter(self, client):
        """days query parameter should be accepted without validation error."""
        resp = await client.get("/api/v1/analytics/summary?days=7")
        # summary endpoint doesn't take days param — 422 only if bad param passthrough
        assert resp.status_code != 422, "Unexpected validation error"


class TestAnalyticsRecent:
    async def test_recent_endpoint_registered(self, client):
        resp = await client.get("/api/v1/analytics/recent")
        assert resp.status_code != 404

    async def test_recent_accepts_limit_param(self, client):
        resp = await client.get("/api/v1/analytics/recent?days=5")
        assert resp.status_code != 422

    async def test_recent_invalid_limit_rejected(self, client):
        # days=0 is below min=1 — should be rejected with 422
        resp = await client.get("/api/v1/analytics/recent?days=0")
        assert resp.status_code == 422


class TestAnalyticsProcessingTimes:
    async def test_processing_times_endpoint_registered(self, client):
        resp = await client.get("/api/v1/analytics/processing-times")
        assert resp.status_code != 404

    async def test_processing_times_no_data(self, client):
        """With empty DB the endpoint should return a graceful error payload, not 500."""
        resp = await client.get("/api/v1/analytics/processing-times")
        # Route returns {"error": "No completed analyses yet"} when empty
        assert resp.status_code in (200, 204)
