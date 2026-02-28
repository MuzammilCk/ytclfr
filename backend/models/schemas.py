"""
models/schemas.py
Pydantic v2 schemas (request bodies, response models, DTOs).
"""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


# ── Shared ────────────────────────────────────────────────────────────────────
class OKResponse(BaseModel):
    ok: bool = True
    message: str = "success"


# ── Auth ──────────────────────────────────────────────────────────────────────
class UserRegisterRequest(BaseModel):
    email: str = Field(..., pattern=r"^[^@]+@[^@]+\.[^@]+$")
    password: str = Field(..., min_length=8)
    display_name: Optional[str] = Field(None, max_length=100)


class UserLoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(BaseModel):
    id: UUID
    email: str
    display_name: Optional[str]
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Analysis submission ────────────────────────────────────────────────────────
class AnalysisRequest(BaseModel):
    url: str = Field(..., description="Full YouTube video URL or short youtu.be link")
    force_reanalysis: bool = Field(
        False, description="If True, bypass cache and rerun even if already analysed"
    )

    @field_validator("url")
    @classmethod
    def validate_youtube_url(cls, v: str) -> str:
        v = v.strip()
        if not (
            "youtube.com/watch" in v
            or "youtu.be/" in v
            or "youtube.com/shorts/" in v
        ):
            raise ValueError("URL must be a valid YouTube video link")
        return v


class BatchAnalysisRequest(BaseModel):
    urls: List[str] = Field(..., min_length=1, max_length=50)
    force_reanalysis: bool = False

    @field_validator("urls")
    @classmethod
    def validate_urls(cls, urls: List[str]) -> List[str]:
        validated = []
        for u in urls:
            u = u.strip()
            if not ("youtube.com/watch" in u or "youtu.be/" in u):
                raise ValueError(f"Invalid YouTube URL: {u}")
            validated.append(u)
        return validated


class AnalysisJobResponse(BaseModel):
    analysis_id: UUID
    status: str
    estimated_seconds: Optional[int] = None
    message: str


# ── Video metadata ─────────────────────────────────────────────────────────────
class VideoMetadata(BaseModel):
    youtube_id: str
    title: str
    channel_name: Optional[str]
    duration_secs: Optional[int]
    view_count: Optional[int]
    like_count: Optional[int]
    thumbnail_url: Optional[str]
    upload_date: Optional[str]
    tags: List[str] = []
    language: Optional[str]


# ── Classification ─────────────────────────────────────────────────────────────
class CategoryScore(BaseModel):
    category: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class ClassificationResult(BaseModel):
    predicted_category: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    all_scores: List[CategoryScore]
    modality_breakdown: Dict[str, float] = {}


# ── Transcript ────────────────────────────────────────────────────────────────
class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: Optional[str] = None


class TranscriptResult(BaseModel):
    full_text: str
    language: str
    segments: List[TranscriptSegment]
    word_count: int


# ── Category-specific outputs ─────────────────────────────────────────────────

# Comedy
class ComedyTimestamp(BaseModel):
    type: str              # "punchline" | "laugh_track" | "scene_change"
    timestamp_secs: float
    description: str


class ComedyOutput(BaseModel):
    transcript: TranscriptResult
    key_moments: List[ComedyTimestamp]
    sentiment_arc: List[Dict[str, float]]    # [{time, score}]
    scene_segments: List[Dict[str, Any]]


# Listicle
class StreamingAvailability(BaseModel):
    flatrate: List[str] = []
    rent: List[str] = []
    buy: List[str] = []


class ListicleItem(BaseModel):
    rank: Optional[int]
    title: str
    description: Optional[str]
    year: Optional[str]
    tmdb_rating: Optional[float]
    tmdb_id: Optional[int]
    poster_url: Optional[str]
    streaming: Optional[StreamingAvailability]
    imdb_url: Optional[str]


class ListicleOutput(BaseModel):
    list_title: str
    items: List[ListicleItem]
    total_count: int


# Music
class SpotifyTrack(BaseModel):
    spotify_id: Optional[str]
    spotify_uri: Optional[str]
    spotify_url: Optional[str]
    preview_url: Optional[str]
    found: bool = False


class MusicTrack(BaseModel):
    rank: Optional[int]
    title: str
    artist: str
    album: Optional[str]
    year: Optional[str]
    genre: Optional[str]
    timestamp_secs: Optional[float]
    spotify: Optional[SpotifyTrack]


class MusicOutput(BaseModel):
    tracks: List[MusicTrack]
    total_count: int
    spotify_playlist_url: Optional[str] = None
    apple_music_search_url: Optional[str] = None


# Educational
class TutorialChapter(BaseModel):
    index: int
    title: str
    start_secs: float
    end_secs: float
    summary: str
    key_concepts: List[str]
    screenshot_url: Optional[str]


class EducationalOutput(BaseModel):
    chapters: List[TutorialChapter]
    key_concepts: List[str]
    summary: str
    transcript: TranscriptResult


# Generic
class GenericOutput(BaseModel):
    transcript: Optional[TranscriptResult]
    summary: str
    key_points: List[str]
    named_entities: Dict[str, List[str]]    # {"PERSON": [...], "ORG": [...]}


# Shopping
class ProductItem(BaseModel):
    name: str
    brand: Optional[str] = None
    category: Optional[str] = None
    frame_timestamps: List[str] = []        # frame file paths where detected
    detection_source: str = "yolo"          # "yolo" | "nlp"
    confidence: Optional[float] = None      # YOLO confidence (0-1)
    search_url: str                         # Google Shopping URL


class ShoppingOutput(BaseModel):
    products: List[ProductItem]
    brand_mentions: List[str] = []
    total_products: int
    summary: str


# ── Full Analysis Result ──────────────────────────────────────────────────────
class AnalysisResult(BaseModel):
    analysis_id: UUID
    video: VideoMetadata
    classification: ClassificationResult
    processing_time_secs: float
    output: ComedyOutput | ListicleOutput | MusicOutput | EducationalOutput | ShoppingOutput | GenericOutput
    created_at: datetime


# ── Export ────────────────────────────────────────────────────────────────────
class ExportRequest(BaseModel):
    analysis_id: UUID
    format: str = Field(..., pattern="^(json|csv|pdf)$")


# ── Spotify playlist creation ─────────────────────────────────────────────────
class SpotifyPlaylistRequest(BaseModel):
    analysis_id: UUID
    playlist_name: Optional[str] = None


class SpotifyPlaylistResponse(BaseModel):
    playlist_id: str
    playlist_url: str
    name: str
    tracks_added: int


# ── Pagination ────────────────────────────────────────────────────────────────
class PaginatedResponse(BaseModel):
    items: List[Any]
    total: int
    page: int
    page_size: int
    has_next: bool


# ── Analytics ─────────────────────────────────────────────────────────────────
class AnalyticsOverviewResponse(BaseModel):
    total_analyses: int
    completed_analyses: int
    failed_analyses: int
    total_videos: int
    category_breakdown: Dict[str, int]       # {"music": 5, "gaming": 3, ...}
    avg_processing_time_secs: Optional[float]
    most_analysed_channel: Optional[str]


class AnalyticsByDateResponse(BaseModel):
    date: str                                # ISO date string, e.g. "2025-01-15"
    analyses_count: int
    categories: Dict[str, int]               # category breakdown for that date
