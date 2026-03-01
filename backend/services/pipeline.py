"""
services/pipeline.py

The main analysis pipeline, implemented as a Celery task.
Orchestrates all services in the correct sequence and writes
results to MongoDB.
"""
from __future__ import annotations

import asyncio
import shutil
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from celery import Celery
from celery.exceptions import SoftTimeLimitExceeded
from celery.signals import worker_process_init
from loguru import logger

from core.config import get_settings
from db.database import get_sync_db
from services.video_processor.downloader import VideoDownloader
from services.video_processor.frame_extractor import FrameExtractor
from services.audio_processor.transcriber import AudioTranscriber
from services.classification.classifier import MultiModalClassifier
from services.extraction.extractors import get_extractor, ShoppingExtractor
from services.integration.tmdb_service import TMDbService
from services.integration.spotify_service import SpotifyService
from services.vision.yolo_detector import YOLODetector

settings = get_settings()

# ── Celery app ────────────────────────────────────────────────────────────────
celery_app = Celery(
    "ytclassifier",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_soft_time_limit=600,
    task_time_limit=720,
    broker_connection_retry_on_startup=True,
)

# ── Module-level model cache (loaded once per worker process) ─────────────────
_models: Dict[str, Any] = {}

# ── Service singletons (one per Celery worker process) ───────────────────────
_downloader: Optional[VideoDownloader] = None
_frame_extractor: Optional[FrameExtractor] = None
_transcriber: Optional[AudioTranscriber] = None
_classifier: Optional[MultiModalClassifier] = None
_tmdb: Optional[TMDbService] = None
_spotify: Optional[SpotifyService] = None
_yolo: Optional[YOLODetector] = None
_sync_mongo_client = None


@worker_process_init.connect
def load_models_on_startup(**kwargs):
    """
    Pre-load all heavy ML models into the module-level _models dict when
    a Celery worker process starts. This avoids cold-start latency on the
    first task and ensures models are shared across tasks in the same process.
    """
    global _models
    logger.info("Worker process init: loading ML models...")

    # Whisper
    try:
        from faster_whisper import WhisperModel
        device = settings.WHISPER_DEVICE
        try:
            _models["whisper"] = WhisperModel(settings.WHISPER_MODEL_SIZE, device=device)
        except Exception:
            logger.warning("CUDA not available for Whisper — falling back to cpu")
            _models["whisper"] = WhisperModel(settings.WHISPER_MODEL_SIZE, device="cpu")
        logger.info(f"Whisper '{settings.WHISPER_MODEL_SIZE}' loaded on {device}")
    except Exception as exc:
        logger.error(f"Failed to load Whisper: {exc}")

    # YOLO
    try:
        from ultralytics import YOLO
        _models["yolo"] = YOLO(settings.YOLO_MODEL_PATH)
        logger.info(f"YOLO '{settings.YOLO_MODEL_PATH}' loaded")
    except Exception as exc:
        logger.error(f"Failed to load YOLO: {exc}")

    logger.info("Worker process init: model loading complete.")


def _get_services():
    global _downloader, _frame_extractor, _transcriber, _classifier, _tmdb, _spotify, _yolo
    if _downloader is None:
        _downloader = VideoDownloader()
        _frame_extractor = FrameExtractor()
        _transcriber = AudioTranscriber()
        _classifier = MultiModalClassifier()
        _tmdb = TMDbService()
        _spotify = SpotifyService()
        _yolo = YOLODetector()
    return _downloader, _frame_extractor, _transcriber, _classifier, _tmdb, _spotify, _yolo


def _update_status(analysis_id: str, status: str, error: Optional[str] = None):
    """
    Update analysis job status in PostgreSQL using the ORM sync session.
    """
    try:
        from sqlalchemy import text
        with get_sync_db() as session:
            if error:
                session.execute(
                    text("UPDATE analyses SET status=:status, error_message=:error WHERE id=:id"),
                    {"status": status, "error": error, "id": analysis_id},
                )
            else:
                session.execute(
                    text("UPDATE analyses SET status=:status WHERE id=:id"),
                    {"status": status, "id": analysis_id},
                )
    except Exception as exc:
        logger.warning(f"Status update failed (non-fatal): {exc}")


def _check_existing(video_url: str) -> Optional[str]:
    """Return an existing completed analysis_id for this URL, or None."""
    try:
        from sqlalchemy import text
        with get_sync_db() as session:
            row = session.execute(
                text(
                    "SELECT a.id FROM analyses a "
                    "JOIN videos v ON a.video_id = v.id "
                    "WHERE v.youtube_id = :url AND a.status = 'complete' "
                    "ORDER BY a.created_at DESC LIMIT 1"
                ),
                {"url": video_url},
            ).fetchone()
            return str(row[0]) if row else None
    except Exception as exc:
        logger.warning(f"Dedup check failed (non-fatal): {exc}")
        return None


@celery_app.task(
    bind=True,
    name="analyse_video",
    time_limit=600,
    soft_time_limit=540,
    autoretry_for=(ConnectionError,),
    retry_backoff=True,
    max_retries=3,
)
def analyse_video(
    self,
    analysis_id: str,
    video_url: str,
    video_id_hint: Optional[str] = None,
    force_reanalysis: bool = False,
) -> Dict[str, Any]:
    """
    Main analysis Celery task.

    Sequence
    ────────
    1. Check for existing completed result (dedup)
    2. Download video + audio
    3. Extract frames
    4. Transcribe audio
    5. Multi-modal classification
    6. Category-specific extraction
    7. External API enrichment (TMDb / Spotify)
    8. Persist results to MongoDB
    9. Cleanup local files (always, in finally)
    """
    t0 = time.perf_counter()
    downloader, frame_extractor, transcriber, classifier, tmdb, spotify, yolo_detector = _get_services()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    temp_dirs = []

    try:
        # ── Dedup check ───────────────────────────────────────────────────────
        if not force_reanalysis:
            existing_id = _check_existing(video_url)
            if existing_id:
                logger.info(f"[{analysis_id}] Returning cached result: {existing_id}")
                return {"analysis_id": existing_id, "cached": True}

        # ── Step 1: Download ──────────────────────────────────────────────────
        _update_status(analysis_id, "downloading")
        logger.info(f"[{analysis_id}] Downloading {video_url}")
        download_result = loop.run_until_complete(downloader.download(video_url))
        video_path = download_result.video_path
        audio_path = download_result.audio_path
        metadata = download_result.metadata
        yt_video_id = download_result.video_id
        if hasattr(download_result, "temp_dir") and download_result.temp_dir:
            temp_dirs.append(download_result.temp_dir)

        # ── Step 2: Frame extraction ──────────────────────────────────────────
        _update_status(analysis_id, "extracting_frames")
        frame_result = loop.run_until_complete(
            frame_extractor.extract(video_path, yt_video_id)
        )

        # ── Step 3: Audio transcription ───────────────────────────────────────
        _update_status(analysis_id, "transcribing")
        transcription = loop.run_until_complete(
            transcriber.transcribe(audio_path, language=None)
        )
        transcript_text = transcription.full_text
        transcript_segments = [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "no_speech_prob": s.no_speech_prob,
            }
            for s in transcription.segments
        ]

        # ── Step 4: Classification ────────────────────────────────────────────
        _update_status(analysis_id, "classifying")
        classification = classifier.predict(
            frame_paths=frame_result.frame_paths,
            transcript=transcript_text,
            title=metadata.get("title", ""),
            description=metadata.get("description", ""),
            tags=metadata.get("tags", []),
        )
        category = classification.predicted_category
        logger.info(
            f"[{analysis_id}] Category: {category} "
            f"(confidence={classification.confidence:.2%})"
        )

        # ── Step 5: Category-specific extraction ──────────────────────────────
        _update_status(analysis_id, "extracting_info")
        extractor = get_extractor(category)

        if category == "shopping" and isinstance(extractor, ShoppingExtractor) and yolo_detector is not None:
            logger.info(f"[{analysis_id}] Running YOLO detection ({len(frame_result.frame_paths)} frames)")
            yolo_detections = loop.run_until_complete(yolo_detector.detect(frame_result.frame_paths))
            extractor.detections = yolo_detections
            logger.info(f"[{analysis_id}] YOLO found {len(yolo_detections)} object(s)")

        extraction = extractor.extract(
            transcript_text=transcript_text,
            segments=transcript_segments,
            metadata=metadata,
            frame_paths=frame_result.frame_paths,
        )

        if category in ["listicle", "educational", "shopping"]:
            from services.vision.ocr_service import OCRService
            ocr = OCRService()
            logger.info(f"[{analysis_id}] Running OCR on frames")
            ocr_results = loop.run_until_complete(ocr.extract_from_frames(frame_result.frame_paths))
            extraction["on_screen_text"] = ocr.aggregate_text(ocr_results)

        # ── Step 6: External API enrichment ───────────────────────────────────
        _update_status(analysis_id, "enriching")

        if category == "listicle" and tmdb.is_available():
            items = extraction.get("items", [])
            enriched_items = loop.run_until_complete(tmdb.enrich_list_items(items))
            extraction["items"] = enriched_items

        if category == "music" and spotify.is_available():
            tracks = extraction.get("tracks", [])
            search_results = loop.run_until_complete(
                asyncio.gather(
                    *[spotify.search_track(t.get("title", ""), t.get("artist", ""))
                      for t in tracks],
                    return_exceptions=True,
                )
            )
            for track, sp_result in zip(tracks, search_results):
                if sp_result and not isinstance(sp_result, Exception):
                    track["spotify"] = {
                        "spotify_id": sp_result.spotify_id,
                        "uri": sp_result.uri,
                        "spotify_url": sp_result.spotify_url,
                        "preview_url": sp_result.preview_url,
                        "found": True,
                    }
                else:
                    track["spotify"] = {"found": False}

        # ── Step 7: Assemble result ───────────────────────────────────────────
        processing_secs = round(time.perf_counter() - t0, 2)
        result = {
            "analysis_id": analysis_id,
            "video": {
                **metadata,
                "language": transcription.language,
            },
            "classification": {
                "predicted_category": category,
                "confidence": classification.confidence,
                "all_scores": classification.all_scores,
                "modality_breakdown": classification.modality_breakdown,
            },
            "transcription": {
                "full_text": transcript_text,
                "language": transcription.language,
                "language_probability": transcription.language_probability,
                "word_count": transcription.word_count,
                "segments": transcript_segments,
            },
            "output": extraction,
            "processing_time_secs": processing_secs,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # ── Step 8: Persist to MongoDB ────────────────────────────────────────
        mongo_ok = _persist_to_mongo(analysis_id, result)
        if not mongo_ok:
            _update_status(
                analysis_id, "failed",
                error="Analysis completed but result could not be saved to MongoDB. "
                      "Check that MongoDB is running."
            )
            return {**result, "mongo_save_failed": True}

        _update_status(analysis_id, "complete")
        logger.info(f"[{analysis_id}] Analysis complete in {processing_secs}s")
        return result

    except SoftTimeLimitExceeded:
        logger.error(f"[{analysis_id}] Soft time limit exceeded")
        _update_status(analysis_id, "failed", error="Task timed out (soft limit)")
        raise
    except Exception as exc:
        logger.exception(f"[{analysis_id}] Analysis failed: {exc}")
        _update_status(analysis_id, "failed", error=str(exc))
        raise
    finally:
        # Always clean up temp dirs regardless of success/failure
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)
        loop.close()


def _persist_to_mongo(analysis_id: str, result: Dict[str, Any]) -> bool:
    """
    Write the full analysis result to MongoDB and store the ObjectId back in Postgres.
    Uses get_sync_db() for the PostgreSQL update instead of raw psycopg2.
    """
    global _sync_mongo_client
    from pymongo import MongoClient
    from sqlalchemy import text
    try:
        if _sync_mongo_client is None:
            _sync_mongo_client = MongoClient(settings.mongodb_url, serverSelectionTimeoutMS=5000)
            
        db = _sync_mongo_client[settings.MONGO_DB]
        doc = {"_analysis_id": analysis_id, **result}
        insert_result = db["analysis_results"].insert_one(doc)
        mongo_id = str(insert_result.inserted_id)

        with get_sync_db() as session:
            session.execute(
                text("UPDATE analyses SET mongo_result_id=:mid, completed_at=NOW() WHERE id=:id"),
                {"mid": mongo_id, "id": analysis_id},
            )
        return True
    except Exception as exc:
        logger.error(f"Failed to persist result to MongoDB: {exc}")
        return False