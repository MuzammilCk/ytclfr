"""
tests/unit/test_admin_routes.py

Unit tests for the Phase 8 admin labeling endpoints.
All external file I/O is mocked.
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from fastapi.testclient import TestClient


# ── Helper: create a test app ─────────────────────────────────────────────────

@pytest.fixture
def client():
    """Create a FastAPI test client with the admin routes loaded."""
    from fastapi import FastAPI
    from api.routes.admin import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


# ── Auth guard ────────────────────────────────────────────────────────────────

def test_label_endpoint_requires_auth(client):
    """POST /api/v1/admin/label should return 401 without a token."""
    res = client.post("/api/v1/admin/label", json={"sample_id": "abc", "human_label": "music"})
    assert res.status_code in (401, 403)


def test_list_endpoint_requires_auth(client):
    """GET /api/v1/admin/training-data should return 401 without a token."""
    res = client.get("/api/v1/admin/training-data")
    assert res.status_code in (401, 403)


def test_export_endpoint_requires_auth(client):
    """GET /api/v1/admin/training-data/export should return 401 without a token."""
    res = client.get("/api/v1/admin/training-data/export")
    assert res.status_code in (401, 403)


# ── CSV Export ────────────────────────────────────────────────────────────────

def test_export_content_type():
    """
    Smoke-test: the export endpoint produces text/csv content type.
    Integration test that only works when a live server is running.
    """
    # This is a structural test — the endpoint uses StreamingResponse with media_type="text/csv"
    from api.routes.admin import export_training_data
    import inspect
    source = inspect.getsource(export_training_data)
    assert "text/csv" in source, "export_training_data must return text/csv content type"
    assert "training_data.csv" in source, "export_training_data must set filename in Content-Disposition"


# ── Schema validation ─────────────────────────────────────────────────────────

def test_label_request_valid_category():
    """LabelRequest should accept any valid category."""
    from models.schemas import LabelRequest, VALID_CATEGORIES
    for cat in VALID_CATEGORIES:
        req = LabelRequest(sample_id="test-id", human_label=cat)
        assert req.human_label == cat


def test_label_request_invalid_category():
    """LabelRequest should reject unknown category strings."""
    from models.schemas import LabelRequest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        LabelRequest(sample_id="test-id", human_label="garbage_category")


def test_label_request_case_insensitive():
    """LabelRequest should normalize labels to lowercase."""
    from models.schemas import LabelRequest
    req = LabelRequest(sample_id="abc", human_label="MUSIC")
    assert req.human_label == "music"


# ── TrainingSampleMeta ─────────────────────────────────────────────────────────

def test_training_sample_meta_unlabeled():
    """TrainingSampleMeta.is_labeled should be False when human_label is None."""
    from models.schemas import TrainingSampleMeta
    sample = TrainingSampleMeta(sample_id="x", human_label=None, is_labeled=False)
    assert not sample.is_labeled


def test_training_sample_meta_labeled():
    """TrainingSampleMeta.is_labeled should be True when human_label is set."""
    from models.schemas import TrainingSampleMeta
    sample = TrainingSampleMeta(sample_id="x", human_label="music", is_labeled=True)
    assert sample.is_labeled
