"""
services/intelligence/router.py

Routes each analysis to the correct intelligence tier.

Tier 1 (TODAY):
    No trained checkpoints exist → always routes to LLMBrain.
    LLMBrain does classification + extraction in one API call.

Tier 2 (Month 6 — activates automatically when checkpoints appear):
    Both checkpoint files exist → run local DistilBERT classifier first.
    If confidence >= threshold: use local classification, call LLM for extraction only.
    If confidence < threshold: fall back to full LLMBrain (classify + extract).

The skeleton for Tier 2 is already present here. When Month 6 training is complete
and the .pth files are placed in backend/checkpoints/, routing upgrades automatically.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from core.config import get_settings
from services.intelligence.llm_brain import LLMBrain, BrainResult
from services.intelligence.training_collector import save_training_sample

settings = get_settings()

# Paths to the trained classifier checkpoints (will not exist until Month 6)
_TEXT_CHECKPOINT = Path("backend/checkpoints/best_text_model.pth")
_FRAME_CHECKPOINT = Path("backend/checkpoints/best_frame_model.pth")


class IntelligenceRouter:
    """
    Routes each analysis to the appropriate intelligence tier.

    Instantiate once per Celery worker process (in load_models_on_startup).
    The router is stateless except for the LLMBrain instance it holds.
    """

    def __init__(self):
        self._brain = LLMBrain(
            api_key=settings.GEMINI_API_KEY,
            model=settings.GEMINI_MODEL,
        )
        self._tier2_text_model = None    # loaded lazily when checkpoints appear
        self._tier2_frame_model = None

    def _has_trained_classifier(self) -> bool:
        """
        Returns True ONLY if BOTH checkpoint files exist.
        A single missing checkpoint means Tier 2 is incomplete → Tier 1 is used.
        """
        text_ok = _TEXT_CHECKPOINT.exists()
        frame_ok = _FRAME_CHECKPOINT.exists()
        if text_ok and frame_ok:
            return True
        if text_ok or frame_ok:
            # One exists but not both — log this so it is noticed during Month 6 deployment
            logger.warning(
                f"Incomplete Tier 2 checkpoints: "
                f"text={'found' if text_ok else 'missing'} "
                f"frame={'found' if frame_ok else 'missing'}. "
                f"Using Tier 1 (LLMBrain)."
            )
        return False

    async def run(
        self,
        metadata: Dict[str, Any],
        frame_ocr_results: List,
        transcript_english: str,
        analysis_id: str,
    ) -> BrainResult:
        """
        Route this analysis to the correct tier and return a BrainResult.

        Args:
            metadata:          Video metadata (title, tags, description, etc.)
            frame_ocr_results: List[FrameOCRData] from ocr_service (all frames, not just content).
            transcript_english: English transcript text (may be empty for silent videos).
            analysis_id:       Unique ID for logging and training data.

        Returns:
            BrainResult with category, confidence, and extracted items.
        """
        title = metadata.get("title", "")
        tags = metadata.get("tags", [])
        description = metadata.get("description", "")

        if self._has_trained_classifier():
            # ── Tier 2 path (Month 6+) ────────────────────────────────────────
            # This skeleton activates automatically when checkpoint files appear.
            logger.info(f"[{analysis_id}] Tier 2 routing: using local classifier")
            brain_result = await self._run_tier2(
                title=title,
                tags=tags,
                description=description,
                frame_ocr_results=frame_ocr_results,
                transcript_english=transcript_english,
                analysis_id=analysis_id,
            )
        else:
            # ── Tier 1 path (TODAY — default) ─────────────────────────────────
            logger.info(f"[{analysis_id}] Tier 1 routing: using LLMBrain (Gemini)")
            brain_result = await self._brain.analyze(
                title=title,
                tags=tags,
                description=description,
                frame_ocr_results=frame_ocr_results,
                transcript_english=transcript_english,
                analysis_id=analysis_id,
            )

        # ── Save training sample (always, on both tiers) ─────────────────────
        try:
            await save_training_sample(
                analysis_id=analysis_id,
                metadata=metadata,
                frame_ocr_results=frame_ocr_results,
                transcript_english=transcript_english,
                brain_result=brain_result,
                training_data_dir=settings.TRAINING_DATA_DIR,
            )
        except Exception as exc:
            # Training data save failure is NEVER fatal
            logger.warning(f"[{analysis_id}] Training collector failed (non-fatal): {exc}")

        return brain_result

    async def _run_tier2(
        self,
        title: str,
        tags: List,
        description: str,
        frame_ocr_results: List,
        transcript_english: str,
        analysis_id: str,
    ) -> BrainResult:
        """
        Tier 2 routing: local DistilBERT classification + LLM extraction on low confidence.
        Skeleton only — will be completed in Month 6.
        """
        # Month 6 TODO: Load DistilBERT from checkpoint, run inference on text features.
        # For now, fall back to Tier 1 (LLMBrain) even when checkpoints exist.
        # This ensures Month 6 partial deployments don't break production.
        logger.info(
            f"[{analysis_id}] Tier 2 skeleton invoked but not yet trained. "
            f"Falling back to LLMBrain."
        )
        return await self._brain.analyze(
            title=title,
            tags=tags,
            description=description,
            frame_ocr_results=frame_ocr_results,
            transcript_english=transcript_english,
            analysis_id=analysis_id,
        )
