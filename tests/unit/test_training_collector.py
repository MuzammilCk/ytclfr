"""
tests/unit/test_training_collector.py

Unit tests for the training data collector.
Verifies correct file creation and required field structure.
"""
import json
import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from services.intelligence.llm_brain import BrainResult


def _make_brain_result(category="music", confidence=0.95):
    return BrainResult(
        category=category,
        confidence=confidence,
        reasoning="Test video with song titles in OCR.",
        items=[{"rank": 1, "title": "Test Song", "artist": "Test Artist"}],
        extraction_source="ocr",
        language="en",
        raw_response='{"category":"music"}',
        input_token_count=200,
        output_token_count=100,
        model_used="gemini-1.5-flash",
    )


@pytest.mark.asyncio
async def test_saves_file_with_required_fields(tmp_path):
    """Training sample must be saved as a valid JSON file with all required fields."""
    from services.intelligence.training_collector import save_training_sample

    result = await save_training_sample(
        analysis_id="test-001",
        metadata={"title": "Test Video", "tags": ["music"], "description": "A test video"},
        frame_ocr_results=[
            {"has_content": True, "cleaned_text": "#1 Test Song - Test Artist", "timestamp_secs": 5.0}
        ],
        transcript_english="",
        brain_result=_make_brain_result(),
        training_data_dir=str(tmp_path),
    )

    assert result is not None
    saved_path = tmp_path / "test-001.json"
    assert saved_path.exists()

    data = json.loads(saved_path.read_text(encoding="utf-8"))

    # Required fields
    assert data["id"] == "test-001"
    assert data["title"] == "Test Video"
    assert data["llm_label"] == "music"
    assert data["llm_confidence"] == 0.95
    assert data["llm_items_count"] == 1
    assert data["llm_model_used"] == "gemini-1.5-flash"
    assert data["llm_extraction_source"] == "ocr"
    assert data["llm_language"] == "en"
    assert data["human_label"] is None
    assert data["human_verified"] is False
    assert data["schema_version"] == "1.1"
    assert "ts" in data   # timestamp


@pytest.mark.asyncio
async def test_creates_directory_if_missing(tmp_path):
    """Collector must create the output directory if it does not exist."""
    from services.intelligence.training_collector import save_training_sample

    nested_dir = str(tmp_path / "nested" / "deeply" / "training_data")
    result = await save_training_sample(
        analysis_id="test-002",
        metadata={"title": "X", "tags": [], "description": ""},
        frame_ocr_results=[],
        transcript_english="hello world",
        brain_result=_make_brain_result(category="educational", confidence=0.8),
        training_data_dir=nested_dir,
    )

    assert result is not None
    assert os.path.exists(result)


@pytest.mark.asyncio
async def test_fallback_result_records_fallback_reason(tmp_path):
    """Fallback BrainResult must have fallback_reason in the saved file."""
    from services.intelligence.training_collector import save_training_sample

    fallback = BrainResult(
        category="unknown",
        confidence=0.0,
        reasoning="LLM unavailable.",
        items=[],
        extraction_source="metadata",
        language="en",
        raw_response="",
        input_token_count=0,
        output_token_count=0,
        model_used="fallback(no_api_key)",
        fallback_reason="no_api_key",
    )

    result = await save_training_sample(
        analysis_id="test-003",
        metadata={"title": "Fallback Test", "tags": [], "description": ""},
        frame_ocr_results=[],
        transcript_english="",
        brain_result=fallback,
        training_data_dir=str(tmp_path),
    )

    assert result is not None
    data = json.loads((tmp_path / "test-003.json").read_text(encoding="utf-8"))
    assert data["llm_fallback_reason"] == "no_api_key"
    assert data["llm_label"] == "unknown"
