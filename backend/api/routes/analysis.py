"""
api/routes/analysis.py

REST API routes for submitting and retrieving video analyses.
"""
from __future__ import annotations

import io
import json
import uuid
from functools import lru_cache
from typing import List, Optional

import orjson
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status, Request
from fastapi.responses import Response, StreamingResponse
from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from db.database import get_db_session, get_mongo_db, get_redis
from db.models import Analysis, JobStatus, Video, VideoCategory
from models.schemas import (
    AnalysisJobResponse,
    AnalysisRequest,
    BatchAnalysisRequest,
    ExportRequest,
    OKResponse,
    PaginatedResponse,
    SpotifyPlaylistRequest,
    SpotifyPlaylistResponse,
)
from services.video_processor.downloader import extract_video_id
# NOTE: services.pipeline is imported lazily inside submit_analysis to avoid
# loading heavy ML libraries (torch, ultralytics, faster-whisper) at module
# import time, which would crash the entire FastAPI app if any are missing.

settings = get_settings()
router = APIRouter(prefix="/api/v1/analyses", tags=["Analysis"])


@lru_cache(maxsize=1)
def _get_spotify_service():
    """Lazy singleton — imported and instantiated only on first call."""
    from services.integration.spotify_service import SpotifyService
    return SpotifyService()


# ── Helper: get analysis or 404 ───────────────────────────────────────────────

async def _get_analysis_or_404(
    analysis_id: uuid.UUID, db: AsyncSession
) -> Analysis:
    result = await db.execute(
        select(Analysis).where(Analysis.id == analysis_id)
    )
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/", response_model=AnalysisJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_analysis(
    body: AnalysisRequest,
    db: AsyncSession = Depends(get_db_session),
    redis=Depends(get_redis),
):
    """
    Submit a YouTube URL for analysis.

    Returns immediately with a job ID. Poll /api/v1/analyses/{id}/status
    to track progress, or fetch the full result once status == 'complete'.
    """
    video_id = extract_video_id(body.url)
    if not video_id:
        raise HTTPException(status_code=422, detail="Could not parse YouTube video ID from URL")

    # ── Check Redis cache (avoid re-processing same video) ────────────────────
    if not body.force_reanalysis:
        cached = await redis.get(f"analysis:complete:{video_id}")
        if cached:
            cached_data = orjson.loads(cached)
            return AnalysisJobResponse(
                analysis_id=uuid.UUID(cached_data["analysis_id"]),
                status="complete",
                message="Returning cached result. Use force_reanalysis=true to reprocess.",
            )

    # ── Upsert Video row ──────────────────────────────────────────────────────
    video_result = await db.execute(select(Video).where(Video.youtube_id == video_id))
    video = video_result.scalar_one_or_none()
    if not video:
        video = Video(youtube_id=video_id, title="Pending…")
        db.add(video)
        await db.flush()

    # ── Create Analysis job row ───────────────────────────────────────────────
    analysis = Analysis(video_id=video.id, status=JobStatus.QUEUED)
    db.add(analysis)
    await db.flush()
    analysis_id = str(analysis.id)

    # ── Dispatch Celery task ────────────────────────────────────────────────────
    # Lazy import: avoid loading torch/ultralytics/faster-whisper at startup
    from services.pipeline import analyse_video as celery_analyse  # noqa: PLC0415
    task = celery_analyse.apply_async(
        kwargs={
            "analysis_id": analysis_id,
            "video_url": body.url,
        },
        task_id=analysis_id,   # use same ID for easy lookup
    )

    analysis.celery_task_id = task.id
    await db.flush()

    # Rough ETA: 30s base + 5s per minute of expected duration
    estimated_secs = 60

    logger.info(f"Analysis job {analysis_id} queued for video {video_id}")
    return AnalysisJobResponse(
        analysis_id=analysis.id,
        status="queued",
        estimated_seconds=estimated_secs,
        message="Analysis job accepted. Poll /status for progress.",
    )


@router.post("/batch", response_model=List[AnalysisJobResponse], status_code=202)
async def submit_batch(
    body: BatchAnalysisRequest,
    db: AsyncSession = Depends(get_db_session),
    redis=Depends(get_redis),
):
    """
    Submit up to 10 videos at once. Returns a list of job statuses.
    """
    results = []
    for url in body.urls:
        try:
            job = await submit_analysis(
                AnalysisRequest(url=url, force_reanalysis=body.force_reanalysis),
                db=db,
                redis=redis,
            )
            results.append(job)
        except HTTPException as exc:
            results.append(
                AnalysisJobResponse(
                    analysis_id=uuid.uuid4(),
                    status="failed",
                    message=f"Error: {exc.detail}",
                )
            )
    return results


@router.get("/{analysis_id}/status")
async def get_status(
    analysis_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """Poll the processing status of an analysis job."""
    analysis = await _get_analysis_or_404(analysis_id, db)
    return {
        "analysis_id": str(analysis.id),
        "status": analysis.status.value,
        "error_message": analysis.error_message,
        "processing_time_secs": analysis.processing_time_secs,
    }


@router.get("/{analysis_id}/result")
async def get_result(
    analysis_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Retrieve the full structured analysis result from MongoDB.
    Returns 404 if not yet complete.
    """
    analysis = await _get_analysis_or_404(analysis_id, db)

    if analysis.status != JobStatus.COMPLETE:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis not yet complete. Current status: {analysis.status.value}",
        )
    if not analysis.mongo_result_id:
        raise HTTPException(status_code=500, detail="Result reference missing")

    mongo_db = get_mongo_db()
    from bson import ObjectId
    doc = await mongo_db["analysis_results"].find_one(
        {"_id": ObjectId(analysis.mongo_result_id)}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Result document not found in store")

    doc["_id"] = str(doc["_id"])
    return doc


@router.get("/", response_model=PaginatedResponse)
async def list_analyses(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    category: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db_session),
):
    """List all submitted analyses (paginated)."""
    q = select(Analysis).options(selectinload(Analysis.video)).order_by(Analysis.created_at.desc())

    if category:
        q = q.join(Video).where(Video.category == category)

    result = await db.execute(q.offset((page - 1) * page_size).limit(page_size))
    analyses = result.scalars().all()

    # Efficient count via SELECT COUNT(*) instead of loading all rows
    count_q = select(func.count()).select_from(Analysis)
    if category:
        count_q = count_q.join(Video).where(Video.category == category)
    total = (await db.execute(count_q)).scalar() or 0

    return PaginatedResponse(
        items=[
            {
                "analysis_id": str(a.id),
                "status": a.status.value,
                "video_id": str(a.video_id),
                "created_at": a.created_at.isoformat(),
                "category": a.video.category.value if a.video and a.video.category else None,
            }
            for a in analyses
        ],
        total=total,
        page=page,
        page_size=page_size,
        has_next=(page * page_size) < total,
    )


@router.post("/export", response_class=StreamingResponse)
async def export_result(
    body: ExportRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """Export analysis result as JSON, CSV, or PDF."""
    analysis = await _get_analysis_or_404(body.analysis_id, db)
    if analysis.status != JobStatus.COMPLETE:
        raise HTTPException(status_code=400, detail="Analysis not yet complete")

    mongo_db = get_mongo_db()
    from bson import ObjectId
    doc = await mongo_db["analysis_results"].find_one(
        {"_id": ObjectId(analysis.mongo_result_id)}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Result not found")

    doc.pop("_id", None)

    if body.format == "json":
        content = orjson.dumps(doc, option=orjson.OPT_INDENT_2)
        return StreamingResponse(
            io.BytesIO(content),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=analysis_{body.analysis_id}.json"},
        )

    elif body.format == "csv":
        import csv
        import io as std_io

        output = std_io.StringIO()
        output_type = doc.get("output", {}).get("type", "generic")
        writer = csv.writer(output)

        if output_type == "listicle":
            writer.writerow(["Rank", "Title", "Year", "Rating", "Streaming", "IMDb URL"])
            for item in doc.get("output", {}).get("items", []):
                streaming = item.get("streaming") or {}
                platforms = ", ".join(streaming.get("flatrate", []))
                writer.writerow([
                    item.get("rank"),
                    item.get("title"),
                    item.get("year"),
                    item.get("tmdb_rating"),
                    platforms,
                    item.get("imdb_url"),
                ])
        elif output_type == "music":
            writer.writerow(["Rank", "Title", "Artist", "Year", "Spotify URL"])
            for track in doc.get("output", {}).get("tracks", []):
                sp = track.get("spotify") or {}
                writer.writerow([
                    track.get("rank"),
                    track.get("title"),
                    track.get("artist"),
                    track.get("year"),
                    sp.get("spotify_url"),
                ])
        elif output_type == "shopping":
            writer.writerow(["Name", "Brand", "Category", "Confidence", "Search URL"])
            for product in doc.get("output", {}).get("products", []):
                writer.writerow([
                    product.get("name"),
                    product.get("brand") or "",
                    product.get("category") or "",
                    product.get("confidence") or "",
                    product.get("search_url") or "",
                ])
        else:
            writer.writerow(["Key", "Value"])
            for k, v in doc.get("output", {}).items():
                writer.writerow([k, str(v)[:200]])

        csv_bytes = output.getvalue().encode()
        return StreamingResponse(
            io.BytesIO(csv_bytes),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=analysis_{body.analysis_id}.csv"},
        )

    elif body.format == "pdf":
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
            from reportlab.lib import colors
        except ImportError:
            raise HTTPException(status_code=501, detail="PDF export not available: install reportlab")

        buf = io.BytesIO()
        doc_pdf = SimpleDocTemplate(buf, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        video_meta = doc.get("video", {})
        story.append(Paragraph(video_meta.get("title", "Analysis Report"), styles["Title"]))
        story.append(Spacer(1, 12))

        cls = doc.get("classification", {})
        story.append(Paragraph(
            f"Category: {cls.get('predicted_category', '')} "
            f"(confidence: {cls.get('confidence', 0):.0%})",
            styles["Heading2"],
        ))
        story.append(Spacer(1, 12))

        output_data = doc.get("output", {})
        output_type = output_data.get("type", "")

        if output_type == "listicle" and output_data.get("items"):
            story.append(Paragraph("Ranked List", styles["Heading2"]))
            story.append(Spacer(1, 6))
            table_data = [["#", "Title", "Rating", "Streaming"]]
            for item in output_data["items"]:
                streaming = item.get("streaming") or {}
                platforms = ", ".join(streaming.get("flatrate", [])[:3])
                table_data.append([
                    str(item.get("rank", "")),
                    item.get("title", ""),
                    str(item.get("tmdb_rating", "") or ""),
                    platforms,
                ])
            t = Table(table_data, colWidths=[30, 250, 60, 150])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f4")]),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(t)

        elif output_type == "music" and output_data.get("tracks"):
            story.append(Paragraph("Track List", styles["Heading2"]))
            story.append(Spacer(1, 6))
            table_data = [["#", "Title", "Artist", "Year", "Spotify URL"]]
            for track in output_data["tracks"]:
                sp = track.get("spotify") or {}
                sp_url = sp.get("spotify_url", "") or ""
                table_data.append([
                    str(track.get("rank", "")),
                    track.get("title", ""),
                    track.get("artist", ""),
                    track.get("year", "") or "",
                    sp_url[:45] + "…" if len(sp_url) > 45 else sp_url,
                ])
            t = Table(table_data, colWidths=[25, 150, 120, 40, 170])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a1a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f4")]),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(t)
            if output_data.get("spotify_playlist_url"):
                story.append(Spacer(1, 8))
                story.append(Paragraph(
                    f"Spotify Playlist: {output_data['spotify_playlist_url']}",
                    styles["Normal"],
                ))

        elif output_type == "educational":
            chapters = output_data.get("chapters", [])
            if chapters:
                story.append(Paragraph("Chapter Breakdown", styles["Heading2"]))
                story.append(Spacer(1, 6))
                table_data = [["#", "Title", "Start", "Key Concepts"]]
                for ch in chapters:
                    start_secs = ch.get("start_secs", 0)
                    mins = int(start_secs // 60)
                    secs = int(start_secs % 60)
                    timestamp = f"{mins}:{secs:02d}"
                    concepts = ", ".join(ch.get("key_concepts", [])[:4])
                    table_data.append([
                        str(ch.get("index", "")),
                        ch.get("title", ""),
                        timestamp,
                        concepts,
                    ])
                t = Table(table_data, colWidths=[25, 180, 45, 250])
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d3349")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef4f8")]),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("WORDWRAP", (1, 1), (1, -1), True),
                ]))
                story.append(t)
                story.append(Spacer(1, 12))

            key_concepts = output_data.get("key_concepts", [])
            if key_concepts:
                story.append(Paragraph("Key Concepts", styles["Heading2"]))
                story.append(Spacer(1, 6))
                concepts_text = " · ".join(key_concepts[:20])
                story.append(Paragraph(concepts_text, styles["Normal"]))

            summary = output_data.get("summary", "")
            if summary:
                story.append(Spacer(1, 10))
                story.append(Paragraph("Summary", styles["Heading2"]))
                story.append(Spacer(1, 4))
                story.append(Paragraph(summary[:800], styles["Normal"]))

        elif output_type == "shopping" and output_data.get("products"):
            story.append(Paragraph("Detected Products", styles["Heading2"]))
            story.append(Spacer(1, 6))
            table_data = [["Name", "Brand", "Category", "Search"]]
            for p in output_data["products"]:
                search_url = p.get("search_url", "")
                table_data.append([
                    p.get("name", ""),
                    p.get("brand", "") or "",
                    p.get("category", "") or "",
                    search_url[:40] + "…" if len(search_url) > 40 else search_url,
                ])
            t = Table(table_data, colWidths=[130, 100, 100, 170])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a2e1a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f0")]),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(t)

        else:
            # Generic fallback — key/value list
            story.append(Paragraph("Extracted Data", styles["Heading2"]))
            story.append(Spacer(1, 6))
            for k, v in output_data.items():
                if k == "type":
                    continue
                val_str = str(v)[:200] if not isinstance(v, (list, dict)) else json.dumps(v)[:200]
                story.append(Paragraph(
                    f"<b>{k}:</b> {val_str}",
                    styles["Normal"],
                ))
                story.append(Spacer(1, 4))

        doc_pdf.build(story)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=analysis_{body.analysis_id}.pdf"},
        )

    raise HTTPException(status_code=400, detail="Unsupported export format")


@router.post("/spotify-playlist")
async def create_spotify_playlist(
    body: SpotifyPlaylistRequest,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Create a Spotify playlist from a completed music video analysis.
    Requires the user to have connected their Spotify account.
    """
    if not _get_spotify_service().is_available():
        raise HTTPException(status_code=503, detail="Spotify integration not configured")

    analysis = await _get_analysis_or_404(body.analysis_id, db)
    if analysis.status != JobStatus.COMPLETE:
        raise HTTPException(status_code=400, detail="Analysis not yet complete")

    mongo_db = get_mongo_db()
    from bson import ObjectId
    doc = await mongo_db["analysis_results"].find_one(
        {"_id": ObjectId(analysis.mongo_result_id)}
    )
    if not doc or doc.get("classification", {}).get("predicted_category") != "music":
        raise HTTPException(
            status_code=400,
            detail="This analysis is not a music compilation video",
        )

    tracks = doc.get("output", {}).get("tracks", [])
    if not tracks:
        raise HTTPException(status_code=400, detail="No tracks found in analysis output")

    # Ensure user is authenticated to fetch their Spotify token
    from api.routes.auth import get_current_user
    current_user = await get_current_user(request, db)

    if not current_user.spotify_access_token:
        raise HTTPException(
            status_code=401,
            detail="Spotify not connected. Visit /api/v1/auth/spotify to connect your account."
        )

    svc = _get_spotify_service()
    try:
        sp_user_id = await svc.get_current_user_id(current_user.spotify_access_token)
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Spotify token invalid or expired: {exc}")

    try:
        result = await svc.create_playlist(
            user_access_token=current_user.spotify_access_token,
            user_spotify_id=sp_user_id,
            playlist_name=f"YT: {doc.get('video', {}).get('title', 'Unknown Title')} Extracts",
            tracks=tracks,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Playlist creation failed: {exc}")

    return {
        "status": "success",
        "playlist_id": result.playlist_id,
        "playlist_url": result.playlist_url,
        "tracks_added": result.tracks_added,
        "tracks_not_found": result.tracks_not_found
    }
