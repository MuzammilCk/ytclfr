"""
services/intelligence/training_collector.py

Auto-saves every brain analysis result as a training sample JSON file.
This is the data engine that will enable Tier 2 (Month 6 — DistilBERT training).

Every analysis generates one file: training_data/{analysis_id}.json
The file is saved asynchronously on a best-effort basis.
Failure to save NEVER blocks or fails the analysis pipeline.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from services.intelligence.llm_brain import BrainResult


async def save_training_sample(
    analysis_id: str,
    metadata: Dict[str, Any],
    frame_ocr_results: List,
    transcript_english: str,
    brain_result: BrainResult,
    training_data_dir: str = "training_data",
) -> Optional[str]:
    """
    Save a training sample JSON for this analysis.

    Args:
        analysis_id:       Unique analysis identifier (used as filename).
        metadata:          Video metadata dict (title, tags, description, etc.)
        frame_ocr_results: List of FrameOCRData from ocr_service.
        transcript_english: English transcript (may be empty for silent videos).
        brain_result:      The BrainResult from IntelligenceRouter.run().
        training_data_dir: Directory to save to (relative to CWD or absolute).

    Returns:
        Path to the saved file, or None if saving failed.
    """
    import asyncio

    def _save_sync() -> Optional[str]:
        try:
            out_dir = Path(training_data_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            # ── Aggregate OCR text for the sample ────────────────────────────
            ocr_lines: List[str] = []
            for frame in frame_ocr_results:
                if hasattr(frame, "has_content"):
                    if frame.has_content and frame.cleaned_text:
                        ocr_lines.append(frame.cleaned_text.strip())
                elif isinstance(frame, dict):
                    if frame.get("has_content") and frame.get("cleaned_text"):
                        ocr_lines.append(frame["cleaned_text"].strip())

            ocr_aggregated = "\n".join(ocr_lines)
            ocr_content_frames = sum(
                1 for f in frame_ocr_results
                if (getattr(f, "has_content", False) if hasattr(f, "has_content") else f.get("has_content", False))
            )
            total_frames = len(frame_ocr_results)

            # ── Transcript preview ───────────────────────────────────────────
            transcript_str = (transcript_english or "").strip()
            transcript_preview = transcript_str[:300] if transcript_str else "(silent)"

            # ── Build sample ─────────────────────────────────────────────────
            sample = {
                "id": analysis_id,
                "ts": datetime.now(timezone.utc).isoformat(),
                "title": metadata.get("title", ""),
                "tags": metadata.get("tags", [])[:30],
                "description_preview": (metadata.get("description") or "")[:400],
                "ocr_aggregated": ocr_aggregated[:5000],   # cap to keep files manageable
                "ocr_frame_count": total_frames,
                "ocr_content_frame_count": ocr_content_frames,
                "transcript_preview": transcript_preview,
                "llm_label": brain_result.category,
                "llm_confidence": round(brain_result.confidence, 4),
                "llm_items_count": len(brain_result.items),
                "llm_extraction_source": brain_result.extraction_source,
                "llm_model_used": brain_result.model_used,
                "llm_language": brain_result.language,
                "llm_reasoning": brain_result.reasoning,
                "llm_fallback_reason": brain_result.fallback_reason,
                "human_label": None,
                "human_verified": False,
                "schema_version": "1.1",
            }

            out_path = out_dir / f"{analysis_id}.json"
            out_path.write_text(
                json.dumps(sample, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.debug(f"[{analysis_id}] Training sample saved → {out_path}")
            return str(out_path)

        except Exception as exc:
            logger.warning(
                f"[{analysis_id}] Training sample save failed (non-fatal): {exc}"
            )
            return None

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _save_sync)
