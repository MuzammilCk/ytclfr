"""
tests/unit/test_llm_brain.py

Unit tests for the LLMBrain intelligence layer.
All Gemini API calls are mocked — no real API key required to run these tests.
"""
import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

# We test the brain module directly
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from services.intelligence.llm_brain import LLMBrain, BrainResult, VALID_CATEGORIES


def _make_brain(api_key: str = "test-key-123") -> LLMBrain:
    return LLMBrain(api_key=api_key, model="gemini-1.5-flash")


def _make_ocr_frame(text: str, ts: float = 0.0):
    """Return a dict that mimics a FrameOCRData for test input."""
    return {
        "has_content": bool(text.strip()),
        "cleaned_text": text,
        "timestamp_secs": ts,
    }


# ── North Star Test ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_north_star_music_video():
    """
    North star: silent music countdown video with 20 songs in OCR
    must produce category=music with 20 track items from OCR alone.
    """
    # Build 20 OCR frames that look like a countdown
    tracks_in_ocr = [
        {"rank": i, "title": f"Song Title {i}", "artist": f"Artist {i}"}
        for i in range(1, 21)
    ]
    gemini_response = json.dumps({
        "category": "music",
        "confidence": 0.97,
        "reasoning": "Video shows numbered song/artist pairs in countdown format.",
        "extraction_source": "ocr",
        "language": "en",
        "items": [
            {
                "rank": t["rank"],
                "title": t["title"],
                "artist": t["artist"],
                "timestamp_secs": float(t["rank"] * 10),
                "raw_ocr": f"#{t['rank']} {t['title']} - {t['artist']}",
            }
            for t in tracks_in_ocr
        ],
    })

    frames = [
        _make_ocr_frame(f"#{i} Song Title {i} - Artist {i}", float(i * 10))
        for i in range(1, 21)
    ]

    brain = _make_brain()
    with patch.object(brain, "_call_gemini_sync", return_value=(gemini_response, 350, 420)):
        result = await brain.analyze(
            title="Top 20 Songs of 2023",
            tags=["music", "top20", "countdown"],
            description="",
            frame_ocr_results=frames,
            transcript_english="",   # silent video
            analysis_id="test-north-star",
        )

    assert result.category == "music"
    assert result.confidence >= 0.9
    assert len(result.items) == 20
    assert result.extraction_source == "ocr"
    assert result.fallback_reason is None
    # Verify first and last track
    assert result.items[0]["title"] == "Song Title 1"
    assert result.items[-1]["title"] == "Song Title 20"


# ── Silent video ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_silent_video_still_extracts():
    """
    Silent video (empty transcript) must still return items from OCR.
    Brain must NOT return unknown just because transcript is empty.
    """
    gemini_response = json.dumps({
        "category": "music",
        "confidence": 0.92,
        "reasoning": "Song titles and artist names visible in OCR frames.",
        "extraction_source": "ocr",
        "language": "en",
        "items": [
            {"rank": 1, "title": "Blinding Lights", "artist": "The Weeknd", "timestamp_secs": 5.0, "raw_ocr": "#1 Blinding Lights - The Weeknd"},
        ],
    })

    brain = _make_brain()
    with patch.object(brain, "_call_gemini_sync", return_value=(gemini_response, 200, 150)):
        result = await brain.analyze(
            title="Top Songs 2020",
            tags=["playlist"],
            description="",
            frame_ocr_results=[_make_ocr_frame("#1 Blinding Lights - The Weeknd", 5.0)],
            transcript_english="",  # silent
            analysis_id="test-silent",
        )

    assert result.category == "music"
    assert len(result.items) == 1
    assert result.items[0]["title"] == "Blinding Lights"
    assert result.fallback_reason is None


# ── JSON parse failure ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_json_parse_failure_returns_fallback():
    """
    Malformed JSON from Gemini must produce a fallback BrainResult,
    never raise an exception to the caller.
    """
    brain = _make_brain()
    with patch.object(brain, "_call_gemini_sync", return_value=("THIS IS NOT JSON {{{{", 100, 20)):
        result = await brain.analyze(
            title="Some Video",
            tags=[],
            description="",
            frame_ocr_results=[],
            transcript_english="Some transcript text",
            analysis_id="test-json-fail",
        )

    assert result.category == "unknown"
    assert result.confidence == 0.0
    assert result.fallback_reason == "json_parse_failed"
    assert result.items == []


# ── Rate limit retry ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_empty_response_returns_fallback():
    """
    Empty response from Gemini must produce a fallback result, not crash.
    """
    brain = _make_brain()
    with patch.object(brain, "_call_gemini_sync", return_value=("", 0, 0)):
        result = await brain.analyze(
            title="Any Video",
            tags=[],
            description="",
            frame_ocr_results=[],
            transcript_english="",
            analysis_id="test-empty-resp",
        )

    assert result.category == "unknown"
    assert result.fallback_reason == "empty_response"


# ── No API key ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_no_api_key_returns_fallback():
    """Brain with empty API key must return fallback without crashing."""
    brain = _make_brain(api_key="")
    result = await brain.analyze(
        title="Any Video",
        tags=[],
        description="",
        frame_ocr_results=[],
        transcript_english="bla bla bla",
        analysis_id="test-no-key",
    )
    assert result.category == "unknown"
    assert result.fallback_reason == "no_api_key"


# ── Category validation ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_invalid_category_coerced_to_unknown():
    """If Gemini returns a made-up category, it must be coerced to 'unknown'."""
    bad_response = json.dumps({
        "category": "TOTALLY_MADE_UP",
        "confidence": 0.99,
        "reasoning": "test",
        "extraction_source": "ocr",
        "language": "en",
        "items": [],
    })
    brain = _make_brain()
    with patch.object(brain, "_call_gemini_sync", return_value=(bad_response, 50, 30)):
        result = await brain.analyze(
            title="X", tags=[], description="", frame_ocr_results=[],
            transcript_english="", analysis_id="test-bad-cat"
        )
    assert result.category == "unknown"


# ── User message construction ─────────────────────────────────────────────────

def test_build_user_message_puts_ocr_first():
    """OCR section must appear before transcript in the user message."""
    brain = _make_brain()
    # Use a transcript with more than 5 words so it is not treated as silent
    long_transcript = "This is a proper transcript with many words to avoid silent detection."
    msg = brain._build_user_message(
        title="Test Video",
        tags=["music"],
        description="A test description",
        frame_ocr_results=[_make_ocr_frame("Song Title Artist Name", 1.5)],
        transcript_english=long_transcript,
    )
    ocr_pos = msg.find("FRAME OCR TEXT")
    transcript_pos = msg.find("AUDIO TRANSCRIPT")
    assert ocr_pos < transcript_pos, "OCR section must appear before transcript section"
    assert "Song Title Artist Name" in msg
    assert "proper transcript" in msg


def test_build_user_message_silent_video():
    """Silent video (empty transcript) must show the silent marker."""
    brain = _make_brain()
    msg = brain._build_user_message(
        title="Silent Music Video",
        tags=[],
        description="",
        frame_ocr_results=[],
        transcript_english="",
    )
    # The silent marker is injected when transcript is empty or under 5 words
    assert "silent" in msg.lower()
