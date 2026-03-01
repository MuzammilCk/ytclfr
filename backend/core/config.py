"""
core/config.py
Centralised settings loaded from environment variables / .env file.
All secrets live here — never hard-code them.
"""
from functools import lru_cache
from pathlib import Path
import json
from typing import List, Optional
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from typing_extensions import Annotated
from loguru import logger


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Use absolute path so .env is found regardless of working directory
        env_file=Path(__file__).resolve().parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────────
    APP_NAME: str = "YouTube Intelligent Classifier"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"          # development | staging | production
    SECRET_KEY: str = "change-me-in-production-use-openssl-rand-hex-32"
    SENTRY_DSN: Optional[str] = None

    @model_validator(mode="after")
    def validate_secret_key(self) -> "Settings":
        if self.ENVIRONMENT == "production" and self.SECRET_KEY == "change-me-in-production-use-openssl-rand-hex-32":
            raise ValueError("SECRET_KEY must be changed in production. Run: openssl rand -hex 32")
        return self

    ALLOWED_ORIGINS: Annotated[List[str], NoDecode] = [
        "http://localhost:3000",
        "http://localhost:5173",
    ]

    # ── Database ─────────────────────────────────────────────────────────────
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "ytclassifier"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"

    # Prefer MONGO_URL (full connection string, e.g. Atlas); fall back to MONGO_URI for legacy configs
    MONGO_URL: Optional[str] = None
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB: str = "ytclassifier_docs"

    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 86400           # 24 h

    # ── Celery ───────────────────────────────────────────────────────────────
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # RabbitMQ (optional alternative broker)
    RABBITMQ_URL: Optional[str] = None

    # ── Storage ──────────────────────────────────────────────────────────────
    DOWNLOAD_DIR: str = "/tmp/ytclassifier/downloads"
    FRAMES_DIR: str = "/tmp/ytclassifier/frames"
    AUDIO_DIR: str = "/tmp/ytclassifier/audio"
    EXPORTS_DIR: str = "/tmp/ytclassifier/exports"
    MAX_VIDEO_DURATION_SECS: int = 3600      # 1 hour hard cap
    CLEANUP_AFTER_SECS: int = 3600           # delete raw files after 1 h

    # ── File size limit ───────────────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = 500              # reject downloads larger than this

    # ── Whisper ──────────────────────────────────────────────────────────────
    WHISPER_MODEL_SIZE: str = "base"         # tiny | base | small | medium | large
    WHISPER_DEVICE: str = "cpu"              # cpu | cuda

    # ── Vision / ML ──────────────────────────────────────────────────────────
    FRAME_SAMPLE_FPS: float = 1.0
    MAX_FRAMES: int = 60                     # cap frames fed to YOLO/EfficientNet
    YOLO_MODEL_PATH: str = "yolov8n.pt"
    YOLO_CONFIDENCE: float = 0.45            # per-box confidence threshold
    CLASSIFICATION_CONFIDENCE_THRESHOLD: float = 0.65
    TORCH_DEVICE: str = "cpu"

    # ── OCR ──────────────────────────────────────────────────────────────────
    OCR_LANG: str = "eng"
    TESSERACT_CMD: Optional[str] = None      # e.g. "C:/Program Files/Tesseract-OCR/tesseract.exe"

    # ── Spotify ──────────────────────────────────────────────────────────────
    SPOTIFY_CLIENT_ID: Optional[str] = None
    SPOTIFY_CLIENT_SECRET: Optional[str] = None
    SPOTIFY_REDIRECT_URI: str = "http://localhost:8000/api/v1/auth/spotify/callback"

    # ── TMDb ─────────────────────────────────────────────────────────────────
    TMDB_API_KEY: Optional[str] = None
    TMDB_BASE_URL: str = "https://api.themoviedb.org/3"
    TMDB_IMAGE_BASE: str = "https://image.tmdb.org/t/p/w500"

    # ── MusicBrainz ──────────────────────────────────────────────────────────
    MUSICBRAINZ_APP_NAME: str = "YTClassifier/1.0"

    # ── Auth / JWT ───────────────────────────────────────────────────────────
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    JWT_ALGORITHM: str = "HS256"

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    RATE_LIMIT_PER_MINUTE: int = 300

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_origins(cls, v):
        """Support JSON array or comma-separated origins from env vars."""
        if isinstance(v, list):
            return [str(origin).strip() for origin in v if str(origin).strip()]

        if isinstance(v, str):
            value = v.strip()
            if not value:
                return []

            # JSON array style: '["http://localhost:3000"]'
            if value.startswith("["):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return [str(origin).strip() for origin in parsed if str(origin).strip()]
                except json.JSONDecodeError:
                    pass

            # CSV style: 'http://a.com,http://b.com'
            return [origin.strip() for origin in value.split(",") if origin.strip()]

        return v

    @model_validator(mode="after")
    def validate_torch_device(self) -> "Settings":
        """Fall back to CPU if CUDA is requested but not available."""
        if self.TORCH_DEVICE == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning(
                        "TORCH_DEVICE=cuda requested but CUDA is not available — falling back to cpu"
                    )
                    self.TORCH_DEVICE = "cpu"
            except ImportError:
                logger.warning("torch not installed; cannot validate TORCH_DEVICE — defaulting to cpu")
                self.TORCH_DEVICE = "cpu"
        return self

    # ── Computed DSNs ─────────────────────────────────────────────────────────

    @property
    def async_database_url(self) -> str:
        """Async DSN for FastAPI routes (asyncpg driver)."""
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def database_url(self) -> str:
        """Sync DSN for Celery tasks / Alembic (psycopg2 driver)."""
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # Keep legacy names as aliases so existing callers don't break
    @property
    def postgres_dsn(self) -> str:
        return self.async_database_url

    @property
    def postgres_dsn_sync(self) -> str:
        return self.database_url

    @property
    def mongodb_url(self) -> str:
        """Returns MONGO_URL if set (e.g. Atlas), otherwise falls back to MONGO_URI (local)."""
        return self.MONGO_URL or self.MONGO_URI

    @property
    def mongodb_db(self) -> str:
        return self.MONGO_DB

    # Legacy alias
    @property
    def mongo_connection_string(self) -> str:
        return self.mongodb_url


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton of Settings."""
    return Settings()