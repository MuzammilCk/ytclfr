"""
services/pipeline.py

The main analysis pipeline, implemented as a Celery task.
Orchestrates all services in the correct sequence and writes
results to MongoDB.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from celery import Celery
from loguru import logger

from core.config import get_settings
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
    worker_prefetch_multiplier=1,   # process one task at a time per worker
    task_soft_time_limit=600,       # 10 min soft limit
    task_time_limit=720,            # 12 min hard limit
)

# Module-level service singletons (one per Celery worker process)
_downloader: Optional[VideoDownloader] = None
_frame_extractor: Optional[FrameExtractor] = None
_transcriber: Optional[AudioTranscriber] = None
_classifier: Optional[MultiModalClassifier] = None
_tmdb: Optional[TMDbService] = None
_spotify: Optional[SpotifyService] = None
_yolo: Optional[YOLODetector] = None


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
    return _downloader, _frame_extractor, _transcriber, _classifier, _tmdb, _spotify


def _update_status(analysis_id: str, status: str, error: Optional[str] = None):
    """
    Update analysis job status in PostgreSQL.
    Runs a raw synchronous DB update to avoid async complexity inside Celery.
    """
    import psycopg2
    try:
        conn = psycopg2.connect(settings.postgres_dsn_sync)
        cur = conn.cursor()
        if error:
            cur.execute(
                "UPDATE analyses SET status=%s, error_message=%s WHERE id=%s",
                (status, error, analysis_id),
            )
        else:
            cur.execute(
                "UPDATE analyses SET status=%s WHERE id=%s",
                (status, analysis_id),
            )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as exc:
        logger.warning(f"Status update failed (non-fatal): {exc}")


@celery_app.task(bind=True, name="analyse_video")
def analyse_video(
    self,
    analysis_id: str,
    video_url: str,
    video_id_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main analysis Celery task.

    Sequence
    ────────
    1. Download video + audio
    2. Extract frames
    3. Transcribe audio
    4. Multi-modal classification
    5. Category-specific extraction
    6. External API enrichment (TMDb / Spotify)
    7. Persist results to MongoDB
    8. Cleanup local files

    Returns the full analysis result dict (also stored in MongoDB).
    """
    t0 = time.perf_counter()
    downloader, frame_extractor, transcriber, classifier, tmdb, spotify = _get_services()

    # Celery tasks run in their own OS thread — create a fresh event loop.
    # asyncio.get_event_loop() is deprecated in Python 3.10+ from a non-main thread.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # ── Step 1: Download ──────────────────────────────────────────────────
        _update_status(analysis_id, "downloading")
        logger.info(f"[{analysis_id}] Downloading {video_url}")
        download_result = loop.run_until_complete(downloader.download(video_url))
        video_path = download_result.video_path
        audio_path = download_result.audio_path
        metadata = download_result.metadata
        yt_video_id = download_result.video_id

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

        # ── Step 5: Category-specific extraction ──────────────────────────────────────
        _update_status(analysis_id, "extracting_info")
        extractor = get_extractor(category)

        # For shopping videos: run YOLO on sampled frames and inject detections.
        if category == "shopping" and isinstance(extractor, ShoppingExtractor) and _yolo is not None:
            logger.info(f"[{analysis_id}] Running YOLO detection ({len(frame_result.frame_paths)} frames)")
            yolo_detections = loop.run_until_complete(_yolo.detect(frame_result.frame_paths))
            extractor.detections = yolo_detections
            logger.info(f"[{analysis_id}] YOLO found {len(yolo_detections)} object(s)")


        extraction = extractor.extract(
            transcript_text=transcript_text,
            segments=transcript_segments,
            metadata=metadata,
            frame_paths=frame_result.frame_paths,
        )

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
        _persist_to_mongo(analysis_id, result)

        # ── Cleanup ───────────────────────────────────────────────────────────
        loop.run_until_complete(downloader.cleanup(yt_video_id))
        frame_extractor.cleanup(yt_video_id)

        _update_status(analysis_id, "complete")
        logger.info(
            f"[{analysis_id}] Analysis complete in {processing_secs}s"
        )
        return result

    except Exception as exc:
        logger.exception(f"[{analysis_id}] Analysis failed: {exc}")
        _update_status(analysis_id, "failed", error=str(exc))
        raise   # Celery will mark task as FAILURE
    finally:
        loop.close()


def _persist_to_mongo(analysis_id: str, result: Dict[str, Any]):
    """Write the full analysis result to MongoDB (synchronous pymongo call)."""
    from pymongo import MongoClient
    try:
        client = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client[settings.MONGO_DB]
        doc = {"_analysis_id": analysis_id, **result}
        insert_result = db["analysis_results"].insert_one(doc)
        mongo_id = str(insert_result.inserted_id)
        client.close()

        # Store the MongoDB ObjectId back in PostgreSQL for fast retrieval
        import psycopg2
        conn = psycopg2.connect(settings.postgres_dsn_sync)
        cur = conn.cursor()
        cur.execute(
            "UPDATE analyses SET mongo_result_id=%s, completed_at=NOW() WHERE id=%s",
            (mongo_id, analysis_id),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as exc:
        logger.error(f"Failed to persist result to MongoDB: {exc}")
