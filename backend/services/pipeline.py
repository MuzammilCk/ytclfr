"""
services/pipeline.py

The main analysis pipeline, implemented as a Celery task.
Orchestrates all services in the correct sequence and writes
results to MongoDB.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
from services.intelligence.router import IntelligenceRouter
from services.intelligence.extraction_mapper import brain_result_to_extraction

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

@worker_process_init.connect
def init_sentry(**kwargs):
    if settings.ENVIRONMENT == "production" and settings.SENTRY_DSN:
        import sentry_sdk
        from sentry_sdk.integrations.celery import CeleryIntegration
        sentry_sdk.init(
            dsn=settings.SENTRY_DSN,
            environment=settings.ENVIRONMENT,
            integrations=[CeleryIntegration()],
            traces_sample_rate=1.0,
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

    # Intelligence router (Gemini LLM Brain — replaces EfficientNet+BERT ensemble)
    try:
        _models["brain_router"] = IntelligenceRouter()
        logger.info("IntelligenceRouter (Gemini brain) initialized")
    except Exception as exc:
        logger.error(f"Failed to initialize IntelligenceRouter: {exc}")

    # Legacy classifier kept for heuristic fallback only
    try:
        _models["classifier"] = MultiModalClassifier()
    except Exception as exc:
        logger.warning(f"Legacy classifier load failed (non-fatal, used for heuristics only): {exc}")

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



def _brain_result_to_extraction(brain_result, category: str) -> dict:
    """
    Convert a BrainResult into the extraction dict format that downstream
    enrichment steps (Spotify, TMDb, etc.) already understand.

    This is the translation layer between the new intelligence layer
    and the existing enrichment pipeline.
    """
    from services.intelligence.llm_brain import BrainResult as _BR
    items = brain_result.items or []
    base = {
        "type": category,
        "extraction_source": brain_result.extraction_source,
        "brain_confidence": brain_result.confidence,
        "brain_model": brain_result.model_used,
    }

    if category == "music":
        # Normalise items to {title, artist, rank, timestamp_secs}
        tracks = []
        for item in items:
            if not isinstance(item, dict):
                continue
            title = item.get("title", "").strip()
            artist = item.get("artist", "").strip()
            if not title:
                continue
            tracks.append({
                "title": title,
                "artist": artist,
                "rank": item.get("rank"),
                "timestamp_secs": item.get("timestamp_secs"),
                "raw_ocr": item.get("raw_ocr", ""),
                "source": brain_result.extraction_source,
            })
        return {**base, "tracks": tracks}

    elif category == "listicle":
        list_items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            title = item.get("title", "").strip()
            if not title:
                continue
            list_items.append({
                "rank": item.get("rank"),
                "title": title,
                "year": item.get("year"),
                "timestamp_secs": item.get("timestamp_secs"),
                "raw_ocr": item.get("raw_ocr", ""),
            })
        return {**base, "items": list_items}

    elif category == "shopping":
        products = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "").strip()
            if not name:
                continue
            products.append({
                "name": name,
                "brand": item.get("brand"),
                "price": item.get("price"),
                "category": item.get("category", ""),
                "timestamp_secs": item.get("timestamp_secs"),
            })
        return {**base, "products": products}

    elif category == "recipe":
        # items is a list-of-one dict with 'ingredients' and 'steps'
        recipe_data = items[0] if items and isinstance(items[0], dict) else {}
        return {
            **base,
            "ingredients": recipe_data.get("ingredients", []),
            "steps": recipe_data.get("steps", []),
        }

    elif category == "educational":
        edu_data = items[0] if items and isinstance(items[0], dict) else {}
        return {
            **base,
            "chapters": edu_data.get("chapters", []),
            "key_concepts": edu_data.get("key_concepts", []),
        }

    else:
        # gaming, vlog, news, review, comedy, unknown
        return {**base, "items": items}


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

    async def _async_pipeline() -> Dict[str, Any]:
        temp_dirs = []
        frames_dir = None
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
            download_result = await downloader.download(video_url)
            video_path = download_result.video_path
            audio_path = download_result.audio_path
            metadata = download_result.metadata
            yt_video_id = download_result.video_id
            if hasattr(download_result, "temp_dir") and download_result.temp_dir:
                temp_dirs.append(download_result.temp_dir)

            # ── Step 2: Frame extraction ──────────────────────────────────────────
            _update_status(analysis_id, "extracting_frames")
            frame_result = await frame_extractor.extract(video_path, yt_video_id)
            if hasattr(frame_result, "temp_dir") and frame_result.temp_dir:
                frames_dir = frame_result.temp_dir
            elif frame_result.frame_paths:
                frames_dir = str(Path(frame_result.frame_paths[0]).parent)

            # ── Steps 3+4: OCR and Transcription (PARALLEL) ───────────────────────
            # OCR runs on ALL frames simultaneously with transcription.
            # OCR is the PRIMARY data source — it must complete before the brain is called.
            _update_status(analysis_id, "processing")
            logger.info(
                f"[{analysis_id}] Running OCR + transcription in parallel "
                f"({len(frame_result.frame_paths)} frames)"
            )

            from services.vision.ocr_service import OCRService
            ocr_service_instance = OCRService()

            async def _run_ocr_safe():
                try:
                    return await ocr_service_instance.extract_from_frames(
                        frame_result.frame_paths,
                        audio_language=None,   # language unknown yet; re-applied below if needed
                    )
                except Exception as exc:
                    logger.warning(f"[{analysis_id}] OCR failed (non-fatal): {exc}")
                    return None

            async def _run_transcription_safe():
                try:
                    return await transcriber.transcribe_with_translation(audio_path, language=None)
                except Exception as exc:
                    logger.error(f"[{analysis_id}] Transcription failed: {exc}")
                    raise

            video_ocr_result, transcription = await asyncio.gather(
                _run_ocr_safe(),
                _run_transcription_safe(),
            )

            # If language was detected, re-run OCR with correct Tesseract lang pack
            # (only when language is non-English AND OCR succeeded first time)
            if (
                video_ocr_result is not None
                and transcription is not None
                and transcription.language not in ("en", None)
                and transcription.language_probability > 0.7
            ):
                logger.info(
                    f"[{analysis_id}] Re-running OCR with language hint: {transcription.language}"
                )
                try:
                    video_ocr_result = await ocr_service_instance.extract_from_frames(
                        frame_result.frame_paths,
                        audio_language=transcription.language,
                    )
                except Exception as exc:
                    logger.warning(f"[{analysis_id}] Language-aware OCR re-run failed: {exc}")

            # ── Prepare transcript vars ───────────────────────────────────────────
            transcript_original = transcription.full_text if transcription else ""
            transcript_text = (
                transcription.full_text_english
                if (transcription and transcription.was_translated)
                else transcript_original
            )
            transcript_segments = [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "no_speech_prob": s.no_speech_prob,
                }
                for s in (
                    (transcription.segments_english if transcription.was_translated else transcription.segments)
                    if transcription else []
                )
            ]
            ocr_text_for_classification = (
                video_ocr_result.aggregated_text if video_ocr_result else ""
            )

            # ── Silent video detection ────────────────────────────────────────────
            word_count = len(transcript_text.split()) if transcript_text else 0
            is_silent = word_count < 20
            if is_silent:
                logger.info(
                    f"[{analysis_id}] Silent/no-narration video detected "
                    f"(word_count={word_count}). "
                    f"OCR is the only data source. "
                    f"Content frames: {video_ocr_result.frames_with_text if video_ocr_result else 0}"
                )

            # ── Step 5: Intelligence Brain (classify + extract) ───────────────────
            # IntelligenceRouter replaces the EfficientNet+BERT+LlmExtractor chain.
            # One call → category + items. Brain handles silent videos via OCR.
            _update_status(analysis_id, "analyzing")
            brain_router = _models.get("brain_router")
            brain_result = None

            if brain_router is not None:
                ocr_frames_for_brain = video_ocr_result.frames if video_ocr_result else []
                brain_result = await brain_router.run(
                    metadata=metadata,
                    frame_ocr_results=ocr_frames_for_brain,
                    transcript_english=transcript_text,
                    analysis_id=analysis_id,
                )
                category = brain_result.category
                logger.info(
                    f"[{analysis_id}] Brain result: category={category} "
                    f"confidence={brain_result.confidence:.2f} "
                    f"items={len(brain_result.items)} "
                    f"model={brain_result.model_used}"
                )
            else:
                logger.warning(
                    f"[{analysis_id}] IntelligenceRouter not in _models. "
                    f"Falling back to legacy heuristic classifier."
                )
                category = "unknown"

            # ── Step 6: Category-specific extraction ──────────────────────────────
            # Brain result drives extraction. Heuristic regex is the hard fallback
            # only when brain fails or returns category='unknown'.
            _update_status(analysis_id, "extracting_info")

            # Build extraction from brain_result when brain succeeded with high confidence
            brain_succeeded = (
                brain_result is not None
                and brain_result.fallback_reason is None
                and brain_result.category != "unknown"
            )

            if brain_succeeded:
                # Map brain items into the extraction format that enrichment expects
                extraction = brain_result_to_extraction(brain_result, category)
            else:
                # Hard fallback: legacy heuristic regex extractors
                logger.info(f"[{analysis_id}] Using heuristic fallback extractors (brain unavailable or unknown)")

                # Try llama.cpp LlmExtractor first (if loaded)
                llm_success = False
                extraction = {}
                if llm_extractor.is_available():
                    try:
                        extraction = await asyncio.to_thread(
                            llm_extractor.extract,
                            category=category,
                            transcript=transcript_text,
                            metadata=metadata,
                            ocr_text=ocr_text_for_classification,
                        )
                        if extraction.get("type") != "error":
                            llm_success = True
                    except Exception as exc:
                        logger.warning(f"[{analysis_id}] LlmExtractor exception: {exc}")

                if not llm_success:
                    extractor = get_extractor(category)
                    if category == "shopping" and isinstance(extractor, ShoppingExtractor) and yolo_detector is not None:
                        logger.info(f"[{analysis_id}] Running YOLO detection")
                        yolo_detections = await yolo_detector.detect(frame_result.frame_paths)
                        extractor.detections = yolo_detections

                    ocr_frames = video_ocr_result.frames if video_ocr_result else []
                    extraction = await asyncio.to_thread(
                        extractor.extract,
                        transcript_text=transcript_text,
                        segments=transcript_segments,
                        metadata=metadata,
                        frame_paths=frame_result.frame_paths,
                        frame_ocr_results=ocr_frames,
                    )

            # ── Step 7: External API enrichment ───────────────────────────────────
            _update_status(analysis_id, "enriching")

            if category == "listicle":
                items = extraction.get("items", [])
                is_book_list = extraction.get("is_book_list", False)
                if is_book_list:
                    # ── Google Books enrichment ────────────────────────────────
                    from services.integration.google_books_service import GoogleBooksService
                    books_svc = GoogleBooksService()
                    async def _enrich_book(item):
                        if not item.get("title"):
                            return item
                        book = await books_svc.search_book(item["title"])
                        if book:
                            import dataclasses
                            item.update(dataclasses.asdict(book))
                            item.pop("found", None)
                        return item
                    enriched_items = await asyncio.gather(*[_enrich_book(i) for i in items], return_exceptions=False)
                    extraction["items"] = list(enriched_items)
                elif tmdb.is_available():
                    enriched_items = await tmdb.enrich_list_items(items)
                    extraction["items"] = enriched_items

            if category == "music" and spotify.is_available():
                tracks = extraction.get("tracks", [])
                search_results = await asyncio.gather(
                    *[spotify.search_track(
                        t.get("title", ""), 
                        t.get("artist", ""), 
                        ocr_raw=t.get("title", "") if t.get("source") == "ocr" else None
                      )
                      for t in tracks],
                    return_exceptions=True,
                )
                
                track_dicts_for_playlist = []
                for track, sp_result in zip(tracks, search_results):
                    if sp_result and not isinstance(sp_result, Exception):
                        track["spotify"] = {
                            "spotify_id": sp_result.spotify_id,
                            "uri": sp_result.uri,
                            "spotify_url": sp_result.spotify_url,
                            "preview_url": sp_result.preview_url,
                            "found": True,
                            "match_confidence": sp_result.match_confidence,
                        }
                        track_dicts_for_playlist.append({
                            "title": track.get("title", ""),
                            "artist": track.get("artist", ""),
                            "uri": sp_result.uri
                        })
                    else:
                        track["spotify"] = {"found": False}
                
                # Check for playlist auto-creation
                extraction["pending_playlist_tracks"] = True
                extraction["spotify_playlist_url"] = None
                
                if len(track_dicts_for_playlist) >= 3:
                    try:
                        with get_sync_db() as session:
                            from db.models import Analysis
                            analysis = session.query(Analysis).filter(Analysis.id == analysis_id).first()
                            
                            if analysis and analysis.user and analysis.user.spotify_access_token:
                                user = analysis.user
                                playlist_name = metadata.get("title", "YouTube Extracted Playlist")[:100]
                                logger.info(f"[{analysis_id}] Auto-creating playlist due to linked Spotify account")
                                
                                user_spotify_id = await spotify.get_current_user_id(user.spotify_access_token)
                                pl_result = await spotify.create_playlist(
                                    user.spotify_access_token, 
                                    user_spotify_id, 
                                    playlist_name, 
                                    track_dicts_for_playlist
                                )
                                extraction["spotify_playlist_url"] = pl_result.playlist_url
                                extraction["pending_playlist_tracks"] = False
                    except Exception as exc:
                        logger.warning(f"[{analysis_id}] Failed to auto-create playlist: {exc}")

            # ── Step 8: Assemble result ───────────────────────────────────────────
            processing_secs = round(time.perf_counter() - t0, 2)
            
            # Serialize frames cleanly for MongoDB
            def _frame_path_to_url(frame_path: str, video_id: str) -> str:
                filename = Path(frame_path).name
                base = settings.PUBLIC_BASE_URL.rstrip("/")
                return f"{base}/frames/{video_id}/{filename}"
                
            serialized_frames = []
            if video_ocr_result and video_ocr_result.frames:
                from dataclasses import asdict
                for f in video_ocr_result.frames:
                    # Drop full file paths to save MongoDB space; keep basename relative to task
                    f_dict = asdict(f)
                    f_dict["frame_path"] = os.path.basename(f.frame_path)
                    f_dict["frame_url"] = _frame_path_to_url(f.frame_path, yt_video_id)
                    serialized_frames.append(f_dict)
                    
            ocr_summary = {
                "total_frames": video_ocr_result.total_frames_processed if video_ocr_result else 0,
                "frames_with_text": video_ocr_result.frames_with_text if video_ocr_result else 0,
                "aggregated_text": ocr_text_for_classification,
            }

            result = {
                "analysis_id": analysis_id,
                "video": {
                    **metadata,
                    "language": transcription.language if transcription else "unknown",
                },
                "classification": {
                    "predicted_category": category,
                    "confidence": brain_result.confidence if brain_result else 0.0,
                    "reasoning": brain_result.reasoning if brain_result else "",
                    "model_used": brain_result.model_used if brain_result else "none",
                    "extraction_source": brain_result.extraction_source if brain_result else "metadata",
                    "fallback_reason": brain_result.fallback_reason if brain_result else "no_brain_router",
                },
                "transcription": {
                    "full_text": transcript_original,
                    "full_text_english": transcription.full_text_english if (transcription and transcription.was_translated) else None,
                    "language": transcription.language if transcription else "unknown",
                    "language_probability": transcription.language_probability if transcription else 0.0,
                    "was_translated": transcription.was_translated if transcription else False,
                    "is_silent": is_silent,
                    "word_count": transcription.word_count if transcription else 0,
                    "segments": transcript_segments,
                },
                "ocr_summary": ocr_summary,
                "frames": serialized_frames,
                "output": extraction,
                "processing_time_secs": processing_secs,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            # Training data is collected inside IntelligenceRouter.run() automatically.
            # The old _save_training_sample call is replaced by training_collector.py.

            # ── Step 9: Persist to MongoDB ────────────────────────────────────────
            mongo_ok = await asyncio.to_thread(_persist_to_mongo, analysis_id, result)
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

        finally:
            # Always clean up temp dirs regardless of success/failure
            for temp_dir in temp_dirs:
                if frames_dir and temp_dir == frames_dir and settings.KEEP_FRAMES_AFTER_ANALYSIS:
                    continue  # Keep the frames dir
                shutil.rmtree(temp_dir, ignore_errors=True)

    try:
        return asyncio.run(_async_pipeline())
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
                        SET category=:cat, classification_confidence=:conf 
                        FROM analyses a 
                        WHERE a.video_id = v.id AND a.id = :id
                    """),
                    {"cat": cat, "conf": conf, "id": analysis_id}
                )
        return True
    except Exception as exc:
        logger.error(f"Failed to persist result to MongoDB: {exc}")
        return False