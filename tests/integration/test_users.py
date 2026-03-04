"""
tests/integration/test_users.py

Integration tests for the Admin User Management endpoints.
Verifies that Role-Based Access Control (RBAC) securely guards
all endpoints against standard non-admin users.
"""
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import AsyncClient, ASGITransport

from main import app
from db.database import get_db_session
from api.routes.auth import get_current_user
from db.models import User, UserRole

def get_mock_user(role=UserRole.USER):
    return User(
        id=uuid.uuid4(),
        email="test@example.com",
        display_name="Test User",
        role=role,
        is_active=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

def _make_mock_db(admin=False):
    """Return a mock AsyncSession that returns a single target user."""
    mock_session = AsyncMock()
    mock_result = MagicMock()
    
    target_user = get_mock_user(role=UserRole.ADMIN if admin else UserRole.USER)
    mock_result.scalar_one_or_none.return_value = target_user
    mock_result.all.return_value = [target_user]
    
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.scalars = AsyncMock(return_value=mock_result)
    mock_session.get = AsyncMock(return_value=target_user)
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()
    mock_session.scalar = AsyncMock(return_value=1)
    
    return mock_session


@pytest.fixture
async def admin_client():
    """Client authenticated as an ADMIN user."""
    mock_session = _make_mock_db(admin=True)
    admin_user = get_mock_user(role=UserRole.ADMIN)

    async def override_get_db():
        yield mock_session

    async def override_get_current_user():
        return admin_user

    app.dependency_overrides[get_db_session] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test", headers={"X-Forwarded-For": "10.0.0.1"}
        ) as ac:
            yield ac
    finally:
        app.dependency_overrides.pop(get_db_session, None)
        app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
async def standard_client():
    """Client authenticated as a standard USER."""
    mock_session = _make_mock_db(admin=False)
    standard_user = get_mock_user(role=UserRole.USER)

    async def override_get_db():
        yield mock_session

    async def override_get_current_user():
        return standard_user

    app.dependency_overrides[get_db_session] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test", headers={"X-Forwarded-For": "10.0.0.2"}
        ) as ac:
            yield ac
    finally:
        app.dependency_overrides.pop(get_db_session, None)
        app.dependency_overrides.pop(get_current_user, None)


class TestAdminUsersRouter:
    async def test_standard_user_cannot_list_users(self, standard_client):
        """A standard user calling GET /api/v1/users must receive 403 Forbidden."""
        resp = await standard_client.get("/api/v1/users")
        assert resp.status_code == 403

    async def test_admin_can_list_users(self, admin_client):
        """An admin calling GET /api/v1/users should succeed."""
        resp = await admin_client.get("/api/v1/users")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "total" in data

    async def test_standard_user_cannot_update_role(self, standard_client):
        """A standard user cannot elevate another user's role."""
        target_id = str(uuid.uuid4())
        resp = await standard_client.patch(f"/api/v1/users/{target_id}/role", json={"role": "admin"})
        assert resp.status_code == 403

    async def test_admin_can_update_role(self, admin_client):
        """An admin can update a user's role."""
        target_id = str(uuid.uuid4())
        resp = await admin_client.patch(f"/api/v1/users/{target_id}/role", json={"role": "admin"})
        assert resp.status_code == 200

    async def test_standard_user_cannot_update_status(self, standard_client):
        """A standard user cannot deactivate an account."""
        target_id = str(uuid.uuid4())
        resp = await standard_client.patch(f"/api/v1/users/{target_id}/status", json={"is_active": False})
        assert resp.status_code == 403

    async def test_admin_can_update_status(self, admin_client):
        """An admin can deactivate an account."""
        target_id = str(uuid.uuid4())
        resp = await admin_client.patch(f"/api/v1/users/{target_id}/status", json={"is_active": False})
        assert resp.status_code == 200
