"""
api/routes/analytics.py

Usage analytics dashboard endpoints.
Returns aggregate statistics about processed videos.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import Integer, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db_session
from db.models import Analysis, JobStatus, Video, VideoCategory

router = APIRouter(prefix="/api/v1/analytics", tags=["Analytics"])


@router.get("/summary")
async def get_summary(db: AsyncSession = Depends(get_db_session)):
    """Return top-level usage statistics."""
    # Total analyses
    total_result = await db.execute(select(func.count()).select_from(Analysis))
    total = total_result.scalar() or 0

    # Completed analyses
    done_result = await db.execute(
        select(func.count()).select_from(Analysis).where(Analysis.status == JobStatus.COMPLETE)
    )
    completed = done_result.scalar() or 0

    # Unique videos
    unique_result = await db.execute(select(func.count()).select_from(Video))
    unique_videos = unique_result.scalar() or 0

    # Average processing time (completed only)
    avg_result = await db.execute(
        select(func.avg(Analysis.processing_time_secs)).where(
            Analysis.status == JobStatus.COMPLETE,
            Analysis.processing_time_secs.isnot(None),
        )
    )
    avg_processing = avg_result.scalar()

    # Category breakdown
    cat_result = await db.execute(
        select(Video.category, func.count().label("count"))
        .where(Video.category.isnot(None))
        .group_by(Video.category)
        .order_by(text("count DESC"))
    )
    category_breakdown = {
        row.category.value if hasattr(row.category, "value") else str(row.category): row.count
        for row in cat_result
    }

    return {
        "total_analyses":           total,
        "completed_analyses":       completed,
        "failed_analyses":          total - completed,
        "unique_videos_processed":  unique_videos,
        "avg_processing_time_secs": round(avg_processing, 2) if avg_processing else None,
        "category_breakdown":       category_breakdown,
        "success_rate":             round(completed / total, 4) if total > 0 else None,
    }


@router.get("/recent")
async def get_recent_activity(
    days: int = Query(7, ge=1, le=90),
    db: AsyncSession = Depends(get_db_session),
):
    """Return per-day analysis counts for the last N days."""
    since = datetime.now(timezone.utc) - timedelta(days=days)

    result = await db.execute(
        select(
            func.date_trunc("day", Analysis.created_at).label("day"),
            func.count().label("total"),
            func.coalesce(
                func.sum((Analysis.status == JobStatus.COMPLETE).cast(Integer)), 0
            ).label("completed"),
        )
        .where(Analysis.created_at >= since)
        .group_by(text("day"))
        .order_by(text("day ASC"))
    )

    return {
        "period_days": days,
        "daily": [
            {
                "date":      row.day.date().isoformat(),
                "total":     row.total,
                "completed": int(row.completed or 0),
            }
            for row in result
        ],
    }


@router.get("/processing-times")
async def get_processing_time_distribution(db: AsyncSession = Depends(get_db_session)):
    """Return processing time percentiles for completed analyses."""
    result = await db.execute(
        select(Analysis.processing_time_secs)
        .where(
            Analysis.status == JobStatus.COMPLETE,
            Analysis.processing_time_secs.isnot(None),
        )
    )
    times = [row[0] for row in result if row[0] is not None]

    if not times:
        return {"error": "No completed analyses yet"}

    times_sorted = sorted(times)
    n = len(times_sorted)

    def percentile(p: float) -> float:
        idx = int(p / 100 * n)
        return round(times_sorted[min(idx, n - 1)], 2)

    return {
        "count":    n,
        "min_secs": round(min(times_sorted), 2),
        "max_secs": round(max(times_sorted), 2),
        "mean_secs": round(sum(times_sorted) / n, 2),
        "p50_secs": percentile(50),
        "p75_secs": percentile(75),
        "p90_secs": percentile(90),
        "p95_secs": percentile(95),
        "p99_secs": percentile(99),
    }
