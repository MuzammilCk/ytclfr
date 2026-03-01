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
from prometheus_client import Counter, Histogram

ytclfr_analyses_total = Counter("ytclfr_analyses_total", "Total analyses processed", ["status", "category"])
ytclfr_analysis_duration_seconds = Histogram(
    "ytclfr_analysis_duration_seconds", 
    "Time spent analyzing full video", 
    buckets=[10, 30, 60, 120, 300, 600]
)

from core.config import get_settings
from db.database import get_sync_db
from services.video_processor.downloader import VideoDownloader
from services.video_processor.frame_extractor import FrameExtractor
from services.audio_processor.transcriber import AudioTranscriber
from services.classification.classifier import MultiModalClassifier
from services.extraction.extractors import get_extractor, ShoppingExtractor
from services.extraction.llm_extractor import LlmExtractor
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

    # Classifier models (Frame & Text)
    try:
        from services.classification.classifier import MultiModalClassifier
        classifier_svc = MultiModalClassifier()
        # This will trigger the fallback loading logic inside those methods,
        # load the weights from disk, and return the initialized eval() model.
        _models["efficientnet"] = classifier_svc._get_frame_model()
        logger.info("FrameClassifier (EfficientNet) loaded into cache")
        
        _models["bert"] = classifier_svc._get_text_model()
        logger.info("TextClassifier (BERT) loaded into cache")
    except Exception as exc:
        logger.error(f"Failed to load classification models: {exc}")

    _models["classifier"] = MultiModalClassifier()
    _models["tmdb"] = TMDbService()
    _models["spotify"] = SpotifyService()
    _models["yolo"] = YOLODetector()
    _models["llm_extractor"] = LlmExtractor()
    logger.info("Worker initialization complete.")


def _get_services() -> Tuple[
    VideoDownloader, FrameExtractor, AudioTranscriber, MultiModalClassifier, TMDbService, SpotifyService, YOLODetector, LlmExtractor
]:
    """Helper to inject dependencies."""
    global _downloader, _frame_extractor, _transcriber, _classifier, _tmdb, _spotify, _yolo
    if _downloader is None:
        _downloader = VideoDownloader()
        _frame_extractor = FrameExtractor()
        _transcriber = AudioTranscriber()
        _classifier = _models["classifier"] # Use cached classifier
        _tmdb = _models["tmdb"] # Use cached TMDB
        _spotify = _models["spotify"] # Use cached Spotify
        _yolo = _models["yolo"] # Use cached YOLO
    return (
        _downloader,
        _frame_extractor,
        _transcriber,
        _classifier,
        _tmdb,
        _spotify,
        _yolo,
        _models["llm_extractor"] # Use cached LLM Extractor
    )


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
    6. Category-specific extraction (LLM priority + RegEx fallback)
    7. External API enrichment (TMDb / Spotify)
    8. Persist results to MongoDB
    9. Cleanup local files (always, in finally)
    """
    t0 = time.perf_counter()
    downloader, frame_extractor, transcriber, classifier, tmdb, spotify, yolo_detector, llm_extractor = _get_services()

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

        # ── Step 3b: Pre-classification OCR ───────────────────────────────────
        ocr_text_for_classification = ""
        try:
            from services.vision.ocr_service import OCRService
            ocr = OCRService()
            logger.info(f"[{analysis_id}] Running lightweight pre-classification OCR")
            ocr_results = loop.run_until_complete(ocr.extract_from_frames(frame_result.frame_paths, max_frames=8))
            ocr_text_for_classification = ocr.aggregate_text(ocr_results)
        except Exception as exc:
            logger.warning(f"[{analysis_id}] Pre-classification OCR failed (non-fatal): {exc}")

        # ── Step 4: Classification ────────────────────────────────────────────
        _update_status(analysis_id, "classifying")
        classification = classifier.predict(
            frame_paths=frame_result.frame_paths,
            transcript=transcript_text,
            title=metadata.get("title", ""),
            description=metadata.get("description", ""),
            tags=metadata.get("tags", []),
            ocr_text=ocr_text_for_classification,
        )
        category = classification.predicted_category
        logger.info(
            f"[{analysis_id}] Category: {category} "
            f"(confidence={classification.confidence:.2%})"
        )

        # ── Step 5: Category-specific extraction ──────────────────────────────
        _update_status(analysis_id, "extracting_info")
        
        # 1. Attempt LLM Extraction first (if available)
        llm_success = False
        extraction = {}
        
        if llm_extractor.is_available():
            logger.info(f"[{analysis_id}] Proceeding with LlmExtractor (llama.cpp)")
            try:
                extraction = llm_extractor.extract(
                    category=category,
                    transcript=transcript_text,
                    metadata=metadata,
                    ocr_text=ocr_text_for_classification
                )
                if extraction.get("type") != "error":
                    llm_success = True
                else:
                    logger.warning(f"[{analysis_id}] LLM Extractor failed: {extraction.get('message')}")
            except Exception as e:
                logger.warning(f"[{analysis_id}] LlmExtractor exception: {e}")
                
        # 2. Fall back to heuristic regex extraction if LLM is unavailable or crashes
        if not llm_success:
            logger.info(f"[{analysis_id}] Proceeding with heuristic fallback extractors")
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

            # Only heuristic extractors need a second pass at OCR because 
            # the LlmExtractor prompt already consumed 'ocr_text_for_classification'
            if category in ["listicle", "educational", "shopping", "unknown"]:
                from services.vision.ocr_service import OCRService
                ocr = OCRService()
                logger.info(f"[{analysis_id}] Running OCR on frames")
                ocr_results = loop.run_until_complete(ocr.extract_from_frames(frame_result.frame_paths))
                ocr_aggregated = ocr.aggregate_text(ocr_results)
                extraction["on_screen_text"] = ocr_aggregated

                if category == "listicle" and ocr_aggregated:
                    logger.info(f"[{analysis_id}] Re-running listicle extraction with combined OCR text")
                    combined_text = transcript_text + "\n" + ocr_aggregated
                    extraction = extractor.extract(
                        transcript_text=combined_text,
                        segments=transcript_segments,
                        metadata=metadata,
                        frame_paths=frame_result.frame_paths,
                    )
                    extraction["on_screen_text"] = ocr_aggregated

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

        # ── Training data collection ──────────────────────────────────────────────
        _save_training_sample(
            analysis_id=analysis_id,
            metadata=metadata,
            transcript_text=transcript_text,
            ocr_text=ocr_text_for_classification,
            category=category,
            confidence=classification.confidence,
            extraction=extraction,
        )

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
        ytclfr_analyses_total.labels(status="success", category=category).inc()
        ytclfr_analysis_duration_seconds.observe(processing_secs)
        return result

    except SoftTimeLimitExceeded:
        logger.error(f"[{analysis_id}] Soft time limit exceeded")
        _update_status(analysis_id, "failed", error="Task timed out (soft limit)")
        ytclfr_analyses_total.labels(status="timeout", category="unknown").inc()
        raise
    except Exception as exc:
        logger.exception(f"[{analysis_id}] Analysis failed: {exc}")
        _update_status(analysis_id, "failed", error=str(exc))
        ytclfr_analyses_total.labels(status="failed", category="unknown").inc()
        raise
    finally:
        # Always clean up temp dirs regardless of success/failure
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)
        loop.close()


def _save_training_sample(
    analysis_id: str,
    metadata: dict,
    transcript_text: str,
    ocr_text: str,
    category: str,
    confidence: float,
    extraction: dict,
) -> None:
    """Async-safe training data logger. Fires-and-forgets to disk."""
    import pathlib
    import json
    import datetime
    try:
        sample = {
            "id": analysis_id,
            "ts": datetime.datetime.utcnow().isoformat(),
            "title": metadata.get("title", ""),
            "tags": metadata.get("tags", []),
            "description": metadata.get("description", "")[:1000] if metadata.get("description") else "",
            "transcript": transcript_text[:500],
            "ocr_text": ocr_text[:300],
            "pipeline_category": category,
            "pipeline_confidence": confidence,
            "pipeline_extraction_type": extraction.get("type", ""),
            "item_count": len(extraction.get("items", extraction.get("tracks", []))),
            "human_label": None,
        }
        out = pathlib.Path("training_data") / f"{analysis_id}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(sample, ensure_ascii=False, indent=2))
    except Exception as exc:
        logger.warning(f"Training sample save failed (non-fatal): {exc}")


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

        cat = result.get("classification", {}).get("predicted_category")
        conf = result.get("classification", {}).get("confidence")

        with get_sync_db() as session:
            # Update analysis tracking row
            session.execute(
                text("UPDATE analyses SET mongo_result_id=:mid, completed_at=NOW() WHERE id=:id"),
                {"mid": mongo_id, "id": analysis_id},
            )
            # Write back category to the video row for direct relational queries
            if cat is not None and conf is not None:
                session.execute(
                    text("""
                        UPDATE videos v 
                        SET category=:cat, confidence=:conf 
                        FROM analyses a 
                        WHERE a.video_id = v.id AND a.id = :id
                    """),
                    {"cat": cat, "conf": conf, "id": analysis_id}
                )
        return True
    except Exception as exc:
        logger.error(f"Failed to persist result to MongoDB: {exc}")
        return False