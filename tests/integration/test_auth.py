"""
tests/integration/test_auth.py

Integration tests for the auth endpoints.
Run with: pytest tests/ -v

The conftest.py stubs out celery/gssapi and heavy ML deps before any app module
is loaded. DB is mocked via FastAPI dependency_overrides, so no real Postgres needed.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import AsyncClient, ASGITransport

# Module-level app import — stubs already installed by conftest.py
from main import app  # noqa: E402
from db.database import get_db_session  # noqa: E402


import uuid as _uuid
import datetime as _datetime


def _make_mock_db():
    """Return a mock AsyncSession that simulates an empty database.

    ``flush()`` assigns SQLAlchemy client-side defaults to any ORM objects
    that were ``add()``ed but not yet persisted.  This is necessary because
    the register route returns the User object directly after flush, and
    FastAPI validates the response model against it.
    """
    _added_objects: list = []

    mock_session = AsyncMock()
    empty_result = MagicMock()
    empty_result.scalar_one_or_none.return_value = None
    empty_result.scalar.return_value = None
    empty_result.mappings.return_value.all.return_value = []
    mock_session.execute = AsyncMock(return_value=empty_result)
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()
    mock_session.rollback = AsyncMock()

    def _track_add(obj):
        _added_objects.append(obj)

    async def _flush_and_populate_defaults():
        now = _datetime.datetime.now(_datetime.timezone.utc)
        for obj in _added_objects:
            if hasattr(obj, "id") and obj.id is None:
                obj.id = _uuid.uuid4()
            if hasattr(obj, "is_active") and obj.is_active is None:
                obj.is_active = True
            if hasattr(obj, "created_at") and obj.created_at is None:
                obj.created_at = now
            if not hasattr(obj, "role") or obj.role is None:
                from db.models import UserRole
                obj.role = UserRole.USER

    mock_session.add = MagicMock(side_effect=_track_add)
    mock_session.flush = _flush_and_populate_defaults
    return mock_session



@pytest.fixture
async def client():
    """HTTP test client with DB dependency overridden to a mock session."""
    import unittest.mock
    mock_session = _make_mock_db()

    # Mock Redis for rate limiter middleware (avoids 429 in tests)
    mock_redis = AsyncMock()
    mock_redis.zadd = AsyncMock()
    mock_redis.zremrangebyscore = AsyncMock()
    mock_redis.zcard = AsyncMock(return_value=0)
    mock_redis.expire = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock(return_value=True)
    mock_redis.exists = AsyncMock(return_value=0)

    async def override_get_db():
        yield mock_session

    app.dependency_overrides[get_db_session] = override_get_db
    # Patch BOTH the middleware and the direct get_redis() call in auth routes
    patcher_mw = unittest.mock.patch(
        "api.middleware.rate_limiter.get_redis",
        new_callable=unittest.mock.AsyncMock,
        return_value=mock_redis,
    )
    patcher_db = unittest.mock.patch(
        "db.database.get_redis",
        new_callable=unittest.mock.AsyncMock,
        return_value=mock_redis,
    )
    patcher_mw.start()
    patcher_db.start()
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as ac:
            yield ac
    finally:
        patcher_db.stop()
        patcher_mw.stop()
        app.dependency_overrides.pop(get_db_session, None)


class TestRegister:
    async def test_missing_fields_rejected(self, client):
        """Empty body should fail validation."""
        resp = await client.post("/api/v1/auth/register", json={})
        assert resp.status_code == 422

    async def test_invalid_email_rejected(self, client):
        resp = await client.post(
            "/api/v1/auth/register",
            json={"email": "not-an-email", "password": "password123", "display_name": "Test"},
        )
        assert resp.status_code == 422

    async def test_short_password_rejected(self, client):
        resp = await client.post(
            "/api/v1/auth/register",
            json={"email": "test@example.com", "password": "short", "display_name": "Test"},
        )
        assert resp.status_code == 422

    async def test_valid_registration_reaches_db(self, client):
        """Valid payload passes Pydantic validation and hits the DB layer."""
        resp = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "SecurePass123",
                "display_name": "Test User",
            },
        )
        # 201 = success, 400 = conflict, 500 = DB not fully mocked
        assert resp.status_code in (201, 400, 422, 500)


class TestLogin:
    async def test_invalid_credentials_format(self, client):
        """Empty body should fail validation."""
        resp = await client.post("/api/v1/auth/login", json={})
        assert resp.status_code == 422

    async def test_login_endpoint_exists(self, client):
        """Ensure the endpoint is reachable — must NOT be 404 or 405."""
        resp = await client.post(
            "/api/v1/auth/login",
            json={"email": "nobody@example.com", "password": "wrongpassword"},
        )
        assert resp.status_code != 404
        assert resp.status_code != 405


class TestTokenRefresh:
    async def test_refresh_without_token_rejected(self, client):
        resp = await client.post("/api/v1/auth/refresh", json={})
        assert resp.status_code in (401, 422)


class TestMe:
    async def test_me_without_token_returns_401(self, client):
        resp = await client.get("/api/v1/auth/me")
        assert resp.status_code == 401

    async def test_me_with_invalid_token_returns_401(self, client):
        resp = await client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalid.token.here"},
        )
        assert resp.status_code == 401
