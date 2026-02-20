"""
tests/conftest.py

Shared pytest fixtures for unit and integration tests.
"""
import asyncio
import uuid
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport


# ── Event loop ────────────────────────────────────────────────────────────────
# NOTE: pytest-asyncio >= 0.21 deprecates overriding event_loop at session
# scope via a plain @pytest.fixture. Use the explicit loop_scope parameter on
# @pytest.mark.asyncio or set asyncio_mode = "auto" in pytest.ini instead.
# The fixture below is kept for backward compatibility with older pytest-asyncio.
@pytest.fixture(scope="session")
def event_loop_policy():
    """Return the default asyncio event-loop policy (session-scoped)."""
    return asyncio.DefaultEventLoopPolicy()


# ── App fixture ────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def app():
    """Import and return the FastAPI app (no real DB connections)."""
    with (
        patch("db.database.create_async_engine"),
        patch("db.database.AsyncIOMotorClient"),
    ):
        from main import app as _app
        yield _app


@pytest.fixture
async def client(app) -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


# ── Sample data fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def sample_video_metadata():
    return {
        "youtube_id": "dQw4w9WgXcQ",
        "title": "Top 10 Best Movies of All Time",
        "channel_name": "Cinema Channel",
        "duration_secs": 720,
        "view_count": 1_500_000,
        "like_count": 45_000,
        "description": "1. The Shawshank Redemption\n2. The Godfather\n3. The Dark Knight\n4. Pulp Fiction\n5. Schindler's List",
        "tags": ["movies", "top10", "ranking", "cinema"],
        "thumbnail_url": "https://img.youtube.com/vi/dQw4w9WgXcQ/maxresdefault.jpg",
        "upload_date": "20240115",
    }


@pytest.fixture
def sample_transcript():
    return {
        "full_text": (
            "Welcome to our countdown of the best movies of all time. "
            "Number 1: The Shawshank Redemption. Number 2: The Godfather. "
            "Number 3: The Dark Knight. Number 4: Pulp Fiction. "
            "Number 5: Schindler's List."
        ),
        "language": "en",
        "language_probability": 0.99,
        "word_count": 47,
        "segments": [
            {"start": 0.0, "end": 5.0, "text": "Welcome to our countdown of the best movies of all time.", "no_speech_prob": 0.0},
            {"start": 5.0, "end": 10.0, "text": "Number 1: The Shawshank Redemption.", "no_speech_prob": 0.0},
            {"start": 10.0, "end": 15.0, "text": "Number 2: The Godfather.", "no_speech_prob": 0.0},
        ],
    }


@pytest.fixture
def sample_music_metadata():
    return {
        "youtube_id": "abc12345678",
        "title": "Best 90s Rock Songs Playlist Mix",
        "channel_name": "Rock Mix Channel",
        "duration_secs": 3600,
        "description": (
            "Bohemian Rhapsody - Queen\n"
            "Smells Like Teen Spirit - Nirvana\n"
            "Hotel California - Eagles\n"
            "Nothing Else Matters - Metallica\n"
        ),
        "tags": ["rock", "90s", "playlist", "mix", "songs"],
    }


@pytest.fixture
def sample_analysis_id():
    return str(uuid.uuid4())


# ── Mock external services ─────────────────────────────────────────────────────

@pytest.fixture
def mock_spotify():
    with patch("services.integration.spotify_service.spotipy.Spotify") as mock:
        instance = MagicMock()
        instance.search.return_value = {
            "tracks": {
                "items": [{
                    "id": "spotify_track_id",
                    "uri": "spotify:track:spotify_track_id",
                    "name": "Bohemian Rhapsody",
                    "artists": [{"name": "Queen"}],
                    "album": {"name": "A Night at the Opera", "release_date": "1975-11-21"},
                    "duration_ms": 354000,
                    "popularity": 85,
                    "preview_url": "https://p.scdn.co/mp3-preview/...",
                    "external_urls": {"spotify": "https://open.spotify.com/track/..."},
                }]
            }
        }
        instance.user_playlist_create.return_value = {
            "id": "playlist123",
            "external_urls": {"spotify": "https://open.spotify.com/playlist/playlist123"},
        }
        mock.return_value = instance
        yield instance


@pytest.fixture
def mock_tmdb():
    with patch("services.integration.tmdb_service.httpx.AsyncClient") as mock:
        client = AsyncMock()
        client.get.return_value = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={
                "results": [{
                    "id": 278,
                    "title": "The Shawshank Redemption",
                    "original_title": "The Shawshank Redemption",
                    "release_date": "1994-09-23",
                    "vote_average": 8.7,
                    "vote_count": 25000,
                    "overview": "Two imprisoned men bond over a number of years.",
                    "poster_path": "/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg",
                    "backdrop_path": "/kXfqcdQKsToO0OUXHcrrNCHDBXO.jpg",
                }]
            })
        )
        mock.return_value.__aenter__ = AsyncMock(return_value=client)
        mock.return_value.__aexit__  = AsyncMock(return_value=False)
        yield client
