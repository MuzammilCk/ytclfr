"""
services/intelligence/llm_brain.py

The Gemini LLM brain that replaces the broken EfficientNet+BERT ensemble.

One call does both classification AND extraction.
OCR text from all frames is the PRIMARY data source.
Audio transcript is SUPPLEMENTARY.

The brain returns a BrainResult containing:
  - category (one of 11 categories)
  - confidence
  - items (category-specific structured data)
  - extraction_source (where data came from)

This is the most important file in the Month 1 implementation.
Every other module depends on this working correctly.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

try:
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted, GoogleAPIError
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed; LLMBrain will run in fallback-only mode")


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class BrainResult:
    category: str                  # music | listicle | shopping | recipe | educational | gaming | vlog | news | review | comedy | unknown
    confidence: float              # 0.0 to 1.0 — honest, not optimistic
    reasoning: str                 # one sentence explaining the classification decision
    items: List[Dict[str, Any]]    # category-specific extracted items
    extraction_source: str         # "ocr" | "transcript" | "both" | "metadata"
    language: str                  # detected language: "en", "ko", "es", etc.
    raw_response: str              # full LLM response text for debugging
    input_token_count: int
    output_token_count: int
    model_used: str
    fallback_reason: Optional[str] = None   # set if this is a fallback result


VALID_CATEGORIES = frozenset({
    "music", "listicle", "shopping", "recipe",
    "educational", "gaming", "vlog", "news",
    "review", "comedy", "unknown",
})

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an AI that analyzes YouTube video content and extracts structured data.

Your input is:
1. VIDEO TITLE and TAGS — context signals
2. FRAME OCR TEXT — text visible on screen, in timestamp order (PRIMARY DATA SOURCE)
3. AUDIO TRANSCRIPT — what was spoken (SUPPLEMENTARY, may be empty for silent videos)
4. VIDEO DESCRIPTION — additional context

Your task:
1. Classify the video into exactly ONE category from this list:
   music | listicle | shopping | recipe | educational | gaming | vlog | news | review | comedy | unknown

2. Extract ALL structured items visible in the OCR text. For music, every song title and artist.
   For listicles, every ranked item. Never invent items not present in the provided text.

3. Return ONLY valid JSON. No markdown. No preamble. No explanation. No ```json tags.
   Start your response with { and end with }

CATEGORY DEFINITIONS:
- music: video shows songs/tracks in a list or countdown (look for artist name + song title patterns)
- listicle: numbered ranking of movies, books, games, places, or any items
- shopping: product reviews, hauls, unboxing, recommendations with product names
- recipe: cooking video with ingredients or steps shown on screen
- educational: tutorial, how-to, lecture, course
- gaming: gameplay, walkthrough, esports, tier list
- vlog: personal daily-life video or travel content
- news: news report, analysis, current events
- review: product/movie/game review or comparison
- comedy: skit, prank, stand-up

EXTRACTION RULES:
- OCR is PRIMARY. Use transcript only to fill gaps OCR misses.
- For music: extract EVERY song visible in OCR, even if no transcript mentions it.
- If OCR shows "#1 Blinding Lights - The Weeknd" extract rank=1, title="Blinding Lights", artist="The Weeknd"
- For silent videos: transcript will be empty — that is fine, use OCR only.
- confidence: 0.0=no idea, 0.5=uncertain, 0.8=confident, 0.95=very confident. Be honest.
- If genuinely unclear: use category="unknown" and confidence < 0.4

OUTPUT JSON SCHEMA (always use this exact structure):
{
  "category": "<one of the 11 categories>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence explaining your classification>",
  "extraction_source": "<ocr|transcript|both|metadata>",
  "language": "<ISO 639-1 code of the video's primary language>",
  "items": <category-specific array — see schemas below>
}

ITEM SCHEMAS BY CATEGORY:

music (key="items", array of track objects):
[{"rank": <int or null>, "title": "<song title>", "artist": "<artist name>", "timestamp_secs": <float>, "raw_ocr": "<original OCR line>"}]

listicle (key="items"):
[{"rank": <int>, "title": "<item title>", "year": "<4-digit year string or null>", "timestamp_secs": <float>, "raw_ocr": "<original OCR line>"}]

shopping (key="items"):
[{"name": "<product name>", "brand": "<brand or null>", "price": "<price string or null>", "category": "<product category>", "timestamp_secs": <float>}]

recipe (key="items"):
{"ingredients": [{"quantity": "<amount>", "unit": "<unit>", "name": "<ingredient>"}], "steps": [{"index": <int>, "text": "<step text>", "timestamp_secs": <float>}]}

educational (key="items"):
{"chapters": [{"title": "<chapter title>", "timestamp_secs": <float>}], "key_concepts": ["<concept>", ...]}

For all other categories (gaming, vlog, news, review, comedy, unknown):
"items": []
"""

# ── LLM Brain ─────────────────────────────────────────────────────────────────

class LLMBrain:
    """
    Gemini-powered brain that classifies and extracts in a single API call.

    Usage:
        brain = LLMBrain(api_key=settings.GEMINI_API_KEY)
        result = await brain.analyze(
            title=..., tags=..., description=...,
            frame_ocr_results=..., transcript_english=...,
            analysis_id=...,
        )
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-flash",
    ):
        self._api_key = api_key
        self._model_name = model
        self._model = None   # initialized lazily in first call to avoid import-time side effects

    def _get_model(self):
        """Lazy-initialize the Gemini model (thread-safe — runs in executor)."""
        if self._model is None:
            if not _GEMINI_AVAILABLE:
                raise RuntimeError(
                    "google-generativeai is not installed. "
                    "Run: pip install google-generativeai>=0.8.0"
                )
            if not self._api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. "
                    "Add it to backend/.env and get a free key at "
                    "https://aistudio.google.com/app/apikey"
                )
            genai.configure(api_key=self._api_key)
            self._model = genai.GenerativeModel(self._model_name)
        return self._model

    def _build_user_message(
        self,
        title: str,
        tags: List[str],
        description: str,
        frame_ocr_results: List,        # List[FrameOCRData] from ocr_service
        transcript_english: str,
    ) -> str:
        """Build the structured user message. OCR frames are always listed first."""

        # ── Video metadata ───────────────────────────────────────────────────
        tags_str = ", ".join(str(t) for t in tags[:20]) if tags else "(none)"
        desc_preview = (description or "")[:400].strip() or "(no description)"

        # ── Frame OCR — PRIMARY DATA SOURCE ─────────────────────────────────
        ocr_lines: List[str] = []
        for frame in frame_ocr_results:
            # Handle both dataclass and dict representations
            if hasattr(frame, "has_content"):
                has_content = frame.has_content
                cleaned = frame.cleaned_text
                ts = frame.timestamp_secs
            elif isinstance(frame, dict):
                has_content = frame.get("has_content", False)
                cleaned = frame.get("cleaned_text", frame.get("ocr_text", ""))
                ts = frame.get("timestamp_secs", 0.0)
            else:
                continue

            if has_content and cleaned and cleaned.strip():
                mins = int(ts) // 60
                secs = int(ts) % 60
                ocr_lines.append(f"[{mins:02d}:{secs:02d}] {cleaned.strip()}")

        if ocr_lines:
            ocr_section = "\n".join(ocr_lines)
        else:
            ocr_section = "(no on-screen text detected)"

        # ── Transcript — SUPPLEMENTARY ───────────────────────────────────────
        transcript_preview = (transcript_english or "").strip()
        if not transcript_preview or len(transcript_preview.split()) < 5:
            transcript_section = "(silent video — no spoken content)"
        else:
            # First 3000 chars is plenty for classification + extraction context
            transcript_section = transcript_preview[:3000]

        return (
            f"VIDEO TITLE: {title or '(no title)'}\n"
            f"TAGS: {tags_str}\n\n"
            f"=== FRAME OCR TEXT (PRIMARY SOURCE) ===\n"
            f"{ocr_section}\n\n"
            f"=== AUDIO TRANSCRIPT (SUPPLEMENTARY) ===\n"
            f"{transcript_section}\n\n"
            f"=== VIDEO DESCRIPTION (CONTEXT) ===\n"
            f"{desc_preview}\n\n"
            f"Analyze the above and return the JSON result."
        )

    def _call_gemini_sync(self, user_message: str) -> tuple[str, int, int]:
        """
        Synchronous Gemini API call. Wrapped in run_in_executor for async use.
        Returns (raw_text, input_tokens, output_tokens).

        Error handling:
        - ResourceExhausted (429): wait 60s, retry once
        - Other GoogleAPIError: re-raise to caller
        - Any exception: re-raise to caller
        """
        model = self._get_model()
        full_prompt = _SYSTEM_PROMPT + "\n\n" + user_message

        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            max_output_tokens=2048,
            temperature=0.1,      # low temperature = consistent structured JSON
        )

        def _do_call():
            return model.generate_content(
                full_prompt,
                generation_config=generation_config,
            )

        # First attempt
        try:
            response = _do_call()
        except ResourceExhausted:
            logger.warning("Gemini ResourceExhausted (429). Waiting 60s before retry...")
            time.sleep(60)
            response = _do_call()   # second attempt — if this also fails, exception propagates

        raw_text = response.text

        # Extract token counts if available
        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0

        return raw_text, input_tokens, output_tokens

    async def analyze(
        self,
        title: str,
        tags: List[str],
        description: str,
        frame_ocr_results: List,
        transcript_english: str,
        analysis_id: str,
    ) -> BrainResult:
        """
        Main entry point. Async wrapper around the synchronous Gemini call.
        NEVER raises — always returns a BrainResult (fallback on any failure).
        """
        if not _GEMINI_AVAILABLE or not self._api_key:
            reason = "not_installed" if not _GEMINI_AVAILABLE else "no_api_key"
            logger.warning(f"[{analysis_id}] LLMBrain unavailable ({reason}). Returning fallback.")
            return self._fallback_result(reason, analysis_id)

        user_message = self._build_user_message(
            title=title,
            tags=tags,
            description=description,
            frame_ocr_results=frame_ocr_results,
            transcript_english=transcript_english,
        )

        logger.info(
            f"[{analysis_id}] Calling Gemini ({self._model_name}) | "
            f"OCR frames: {len(frame_ocr_results)} | "
            f"Transcript words: {len((transcript_english or '').split())}"
        )

        try:
            loop = asyncio.get_event_loop()
            raw_text, input_tokens, output_tokens = await loop.run_in_executor(
                None, self._call_gemini_sync, user_message
            )
            logger.info(
                f"[{analysis_id}] Gemini response received "
                f"(in={input_tokens} out={output_tokens} tokens)"
            )
            result = self._parse_response(raw_text, analysis_id)
            result.input_token_count = input_tokens
            result.output_token_count = output_tokens
            return result

        except GoogleAPIError as exc:
            logger.error(f"[{analysis_id}] Gemini API error (non-retryable): {exc}")
            return self._fallback_result(f"api_error: {type(exc).__name__}", analysis_id)
        except Exception as exc:
            logger.error(f"[{analysis_id}] Unexpected LLMBrain error: {exc}")
            return self._fallback_result(f"unexpected: {type(exc).__name__}", analysis_id)

    def _parse_response(self, raw_text: str, analysis_id: str) -> BrainResult:
        """
        Parse Gemini's JSON response into a BrainResult.
        If parsing fails at any stage, returns a fallback result.
        """
        if not raw_text or not raw_text.strip():
            logger.error(f"[{analysis_id}] Brain returned empty response")
            return self._fallback_result("empty_response", analysis_id)

        # Strip any accidental markdown fences (shouldn't happen with mime type set, but be safe)
        clean = raw_text.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:json)?\s*", "", clean)
            clean = re.sub(r"\s*```$", "", clean)
            clean = clean.strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError as exc:
            logger.error(
                f"[{analysis_id}] Brain JSON parse failed: {exc}. "
                f"Raw (first 300 chars): {raw_text[:300]}"
            )
            return self._fallback_result("json_parse_failed", analysis_id, raw_response=raw_text)

        # Validate and normalise category
        category = str(data.get("category", "unknown")).lower().strip()
        if category not in VALID_CATEGORIES:
            logger.warning(
                f"[{analysis_id}] Brain returned unknown category '{category}', "
                f"treating as 'unknown'"
            )
            category = "unknown"

        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))   # clamp to [0, 1]

        reasoning = str(data.get("reasoning", ""))[:500]
        extraction_source = str(data.get("extraction_source", "ocr"))
        language = str(data.get("language", "en"))

        # Extract items — the schema varies by category
        raw_items = data.get("items", [])

        # For recipe, items is a dict not a list — normalise to list-of-one for uniform handling
        if isinstance(raw_items, dict):
            items = [raw_items]
        elif isinstance(raw_items, list):
            items = raw_items
        else:
            items = []

        logger.info(
            f"[{analysis_id}] Brain result: category={category} "
            f"confidence={confidence:.2f} items={len(items)} "
            f"source={extraction_source} lang={language}"
        )

        return BrainResult(
            category=category,
            confidence=confidence,
            reasoning=reasoning,
            items=items,
            extraction_source=extraction_source,
            language=language,
            raw_response=raw_text,
            input_token_count=0,    # updated by caller after parse
            output_token_count=0,
            model_used=self._model_name,
        )

    def _fallback_result(
        self,
        reason: str,
        analysis_id: str,
        raw_response: str = "",
    ) -> BrainResult:
        """
        Returns a safe fallback BrainResult when the LLM call fails.
        The pipeline will fall through to heuristic extractors.
        """
        logger.warning(f"[{analysis_id}] Using fallback BrainResult. Reason: {reason}")
        return BrainResult(
            category="unknown",
            confidence=0.0,
            reasoning=f"LLM brain unavailable or failed ({reason}). Heuristic fallback active.",
            items=[],
            extraction_source="metadata",
            language="en",
            raw_response=raw_response,
            input_token_count=0,
            output_token_count=0,
            model_used=f"fallback({reason})",
            fallback_reason=reason,
        )
