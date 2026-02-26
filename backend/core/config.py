"""
core/config.py
Centralised settings loaded from environment variables / .env file.
All secrets live here — never hard-code them.
"""
from functools import lru_cache
import json
from typing import List, Optional
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from typing_extensions import Annotated


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="../.env",
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
    ALLOWED_ORIGINS: Annotated[List[str], NoDecode] = ["http://localhost:3000"]

    # ── Database ─────────────────────────────────────────────────────────────
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "ytclassifier"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"

    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB: str = "ytclassifier_docs"

    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 86400           # 24 h

    # ── Celery ───────────────────────────────────────────────────────────────
    CELERY_BROKER_URL: str = "amqp://guest:guest@localhost:5672//"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # ── Storage ──────────────────────────────────────────────────────────────
    DOWNLOAD_DIR: str = "/tmp/ytclassifier/downloads"
    FRAMES_DIR: str = "/tmp/ytclassifier/frames"
    AUDIO_DIR: str = "/tmp/ytclassifier/audio"
    EXPORTS_DIR: str = "/tmp/ytclassifier/exports"
    MAX_VIDEO_DURATION_SECS: int = 3600      # 1 hour hard cap
    CLEANUP_AFTER_SECS: int = 3600           # delete raw files after 1 h

    # ── Whisper ──────────────────────────────────────────────────────────────
    WHISPER_MODEL_SIZE: str = "base"         # tiny | base | small | medium | large
    WHISPER_DEVICE: str = "cpu"              # cpu | cuda

    # ── Vision / ML ──────────────────────────────────────────────────────────
    FRAME_SAMPLE_FPS: float = 1.0
    YOLO_MODEL_PATH: str = "yolov8n.pt"
    CLASSIFICATION_CONFIDENCE_THRESHOLD: float = 0.65
    TORCH_DEVICE: str = "cpu"

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
    RATE_LIMIT_PER_MINUTE: int = 20

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

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def postgres_dsn_sync(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton of Settings."""
    return Settings()
