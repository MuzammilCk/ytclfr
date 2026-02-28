"""
db/models.py
SQLAlchemy ORM models for the relational (PostgreSQL) layer.
Document-heavy payloads (transcripts, frame data, extraction results)
live in MongoDB; only structured metadata lives here.
"""
import uuid
from datetime import datetime, timezone
from enum import Enum as PyEnum
from typing import Optional

from sqlalchemy import (
    Boolean, DateTime, Enum, Float, ForeignKey, Integer,
    String, Text, UniqueConstraint, func, text,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.database import Base


def utcnow():
    return datetime.now(timezone.utc)


# ── Enums ─────────────────────────────────────────────────────────────────────
class VideoCategory(str, PyEnum):
    COMEDY = "comedy"
    LISTICLE = "listicle"
    MUSIC = "music"
    EDUCATIONAL = "educational"
    SHOPPING = "shopping"
    NEWS = "news"
    REVIEW = "review"
    GAMING = "gaming"
    VLOG = "vlog"
    UNKNOWN = "unknown"


class JobStatus(str, PyEnum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    EXTRACTING_FRAMES = "extracting_frames"
    TRANSCRIBING = "transcribing"
    CLASSIFYING = "classifying"
    EXTRACTING_INFO = "extracting_info"
    ENRICHING = "enriching"
    COMPLETE = "complete"
    FAILED = "failed"


# ── User ──────────────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(128), nullable=False)
    display_name: Mapped[Optional[str]] = mapped_column(String(100))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    spotify_access_token: Mapped[Optional[str]] = mapped_column(Text)
    spotify_refresh_token: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    analyses: Mapped[list["Analysis"]] = relationship("Analysis", back_populates="user")


# ── Video ─────────────────────────────────────────────────────────────────────
class Video(Base):
    __tablename__ = "videos"
    __table_args__ = (UniqueConstraint("youtube_id"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    youtube_id: Mapped[str] = mapped_column(String(20), unique=True, index=True, nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String(500))
    channel_name: Mapped[Optional[str]] = mapped_column(String(200))
    duration_secs: Mapped[Optional[int]] = mapped_column(Integer)
    view_count: Mapped[Optional[int]] = mapped_column(Integer)
    like_count: Mapped[Optional[int]] = mapped_column(Integer)
    description_preview: Mapped[Optional[str]] = mapped_column(Text)   # first 500 chars
    tags: Mapped[Optional[list]] = mapped_column(JSONB, default=list)
    thumbnail_url: Mapped[Optional[str]] = mapped_column(String(1000))
    upload_date: Mapped[Optional[str]] = mapped_column(String(20))
    language: Mapped[Optional[str]] = mapped_column(String(10))
    category: Mapped[Optional[VideoCategory]] = mapped_column(
    Enum(VideoCategory, values_callable=lambda obj: [e.value for e in obj]),
    default=VideoCategory.UNKNOWN
    )
    classification_confidence: Mapped[Optional[float]] = mapped_column(Float)
    mongo_analysis_id: Mapped[Optional[str]] = mapped_column(String(24))  # ObjectId ref
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    analyses: Mapped[list["Analysis"]] = relationship("Analysis", back_populates="video")


# ── Analysis (job) ────────────────────────────────────────────────────────────
class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    video_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[JobStatus] = mapped_column(
    Enum(JobStatus, values_callable=lambda obj: [e.value for e in obj]),
    default=JobStatus.QUEUED, index=True
    )
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(64))
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    processing_time_secs: Mapped[Optional[float]] = mapped_column(Float)
    mongo_result_id: Mapped[Optional[str]] = mapped_column(String(24))  # ObjectId ref
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    user: Mapped[Optional["User"]] = relationship("User", back_populates="analyses")
    video: Mapped["Video"] = relationship("Video", back_populates="analyses")


# ── Spotify Playlist ──────────────────────────────────────────────────────────
class SpotifyPlaylist(Base):
    __tablename__ = "spotify_playlists"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("analyses.id", ondelete="CASCADE")
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    spotify_playlist_id: Mapped[str] = mapped_column(String(100))
    playlist_url: Mapped[str] = mapped_column(String(500))
    name: Mapped[str] = mapped_column(String(200))
    tracks_added: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
