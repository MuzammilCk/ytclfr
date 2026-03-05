"""
tests/unit/test_intelligence_router.py

Unit tests for IntelligenceRouter tier routing logic.
All I/O is mocked — no real Gemini calls made.
"""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))


def _make_router():
    """Create an IntelligenceRouter with a mocked brain."""
    with patch("core.config.get_settings") as mock_cfg:
        mock_cfg.return_value = MagicMock(
            GEMINI_API_KEY="test-key",
            GEMINI_MODEL="gemini-1.5-flash",
            BRAIN_CONFIDENCE_THRESHOLD=0.85,
            TRAINING_DATA_DIR="training_data",
        )
        from services.intelligence.router import IntelligenceRouter
        router = IntelligenceRouter()
    return router


# ── Tier routing ──────────────────────────────────────────────────────────────

def test_has_trained_classifier_false_when_no_checkpoints(tmp_path):
    """
    _has_trained_classifier() must return False when checkpoint files don't exist.
    """
    with patch("services.intelligence.router._TEXT_CHECKPOINT", tmp_path / "missing_text.pth"), \
         patch("services.intelligence.router._FRAME_CHECKPOINT", tmp_path / "missing_frame.pth"):
        from services.intelligence.router import IntelligenceRouter
        router = IntelligenceRouter.__new__(IntelligenceRouter)
        router._brain = MagicMock()
        router._tier2_text_model = None
        router._tier2_frame_model = None
        assert router._has_trained_classifier() is False


def test_has_trained_classifier_false_when_only_one_checkpoint(tmp_path):
    """
    _has_trained_classifier() must return False if only ONE checkpoint exists.
    Both must be present for Tier 2 to activate.
    """
    text_ckpt = tmp_path / "best_text_model.pth"
    text_ckpt.write_bytes(b"fake")  # exists
    frame_ckpt = tmp_path / "missing_frame.pth"  # does not exist

    with patch("services.intelligence.router._TEXT_CHECKPOINT", text_ckpt), \
         patch("services.intelligence.router._FRAME_CHECKPOINT", frame_ckpt):
        from services.intelligence.router import IntelligenceRouter
        router = IntelligenceRouter.__new__(IntelligenceRouter)
        router._brain = MagicMock()
        router._tier2_text_model = None
        router._tier2_frame_model = None
        assert router._has_trained_classifier() is False


@pytest.mark.asyncio
async def test_router_calls_llm_brain_when_no_checkpoints(tmp_path):
    """
    When no checkpoints exist, router must call LLMBrain.analyze().
    """
    from services.intelligence.llm_brain import BrainResult

    fake_result = BrainResult(
        category="music",
        confidence=0.95,
        reasoning="Test result",
        items=[{"rank": 1, "title": "Test Song", "artist": "Test Artist"}],
        extraction_source="ocr",
        language="en",
        raw_response="{}",
        input_token_count=100,
        output_token_count=50,
        model_used="gemini-1.5-flash",
    )

    with patch("services.intelligence.router._TEXT_CHECKPOINT", tmp_path / "missing.pth"), \
         patch("services.intelligence.router._FRAME_CHECKPOINT", tmp_path / "missing2.pth"), \
         patch("services.intelligence.training_collector.save_training_sample", new_callable=AsyncMock) as mock_save:

        from services.intelligence.router import IntelligenceRouter
        router = IntelligenceRouter.__new__(IntelligenceRouter)
        router._brain = AsyncMock()
        router._brain.analyze = AsyncMock(return_value=fake_result)
        router._tier2_text_model = None
        router._tier2_frame_model = None

        result = await router.run(
            metadata={"title": "Top Songs", "tags": [], "description": ""},
            frame_ocr_results=[],
            transcript_english="",
            analysis_id="test-router-001",
        )

    router._brain.analyze.assert_called_once()
    assert result.category == "music"
    assert result.confidence == 0.95
