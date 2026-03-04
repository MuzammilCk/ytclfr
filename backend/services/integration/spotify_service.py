"""
services/integration/spotify_service.py

Spotify Web API integration via the spotipy library.

Features
────────
- Track search with artist + title disambiguation
- Playlist creation with bulk track insertion
- OAuth token refresh handling
- Graceful fallback when tracks are not found
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import spotipy
from loguru import logger
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth

from core.config import get_settings

settings = get_settings()


import re

@dataclass
class TrackInfo:
    spotify_id: str
    uri: str
    name: str
    artist: str
    album: str
    release_year: Optional[str]
    duration_ms: int
    popularity: int
    preview_url: Optional[str]
    spotify_url: str
    match_confidence: str = "exact"
    found: bool = True


@dataclass
class PlaylistCreationResult:
    playlist_id: str
    playlist_url: str
    name: str
    tracks_added: int
    tracks_not_found: List[str] = field(default_factory=list)


def _build_client_credentials_client() -> Optional[spotipy.Spotify]:
    """
    Creates a Spotify client using Client Credentials flow (no user login required).
    Suitable for read-only operations (search).
    """
    if not settings.SPOTIFY_CLIENT_ID or not settings.SPOTIFY_CLIENT_SECRET:
        logger.warning("Spotify credentials not configured; Spotify integration disabled")
        return None
    return spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(
            client_id=settings.SPOTIFY_CLIENT_ID,
            client_secret=settings.SPOTIFY_CLIENT_SECRET,
        )
    )


def _build_oauth_client(access_token: str) -> spotipy.Spotify:
    """
    Creates a Spotify client using a user's OAuth access token.
    Required for playlist creation (write operation).
    """
    return spotipy.Spotify(auth=access_token)


def _clean_for_search(text: str) -> str:
    """Removes rank numbers, years, and common noise decorators from track/artist names."""
    if not text:
        return ""
    # Remove #1, No. 5, 20.
    t = re.sub(r'^(?:#|No\.?)\s*\d+[\.\)]?\s*', '', text, flags=re.IGNORECASE)
    t = re.sub(r'^\d+[\.\)]\s+', '', t)
    # Remove (2020), [1994]
    t = re.sub(r'[\(\[]\d{4}[\)\]]', '', t)
    # Remove (feat. artist) or ft. artist
    t = re.sub(r'(?i)\(?(?:feat\.|ft\.)[^)]*\)?', '', t)
    # Remove smart quotes and other non-standard chars
    t = t.replace('"', '').replace("'", "").replace("“", "").replace("”", "").replace("’", "").replace("‘", "")
    t = t.replace("—", "-").replace("|", " ")
    return t.strip()


class SpotifyService:
    """
    Wraps spotipy to provide async-friendly track search and playlist creation.
    CPU-bound Spotify I/O is offloaded to a thread executor.
    """

    def __init__(self):
        self._client = _build_client_credentials_client()

    def is_available(self) -> bool:
        return self._client is not None

    # ── Public API ─────────────────────────────────────────────────────────────

    async def search_track(
        self, title: str, artist: str, ocr_raw: Optional[str] = None
    ) -> Optional[TrackInfo]:
        if not self._client:
            return None
        return await asyncio.get_running_loop().run_in_executor(
            None, self._search_track_sync, title, artist, ocr_raw
        )

    async def create_playlist(
        self,
        user_access_token: str,
        user_spotify_id: str,
        playlist_name: str,
        tracks: List[Dict[str, Optional[str]]],   # list of {title, artist}
    ) -> PlaylistCreationResult:
        """
        Search for each track and add found ones to a new Spotify playlist.

        Args:
            user_access_token: OAuth access token of the authenticated user.
            user_spotify_id:   Spotify user ID (from /me endpoint).
            playlist_name:     Display name for the new playlist.
            tracks:            List of {title, artist} dicts.

        Returns:
            PlaylistCreationResult with playlist URL and track count.
        """
        return await asyncio.get_running_loop().run_in_executor(
            None,
            self._create_playlist_sync,
            user_access_token,
            user_spotify_id,
            playlist_name,
            tracks,
        )

    async def get_current_user_id(self, user_access_token: str) -> str:
        """Return the Spotify user ID for the given OAuth token."""
        def _get():
            client = _build_oauth_client(user_access_token)
            return client.current_user()["id"]

        return await asyncio.get_running_loop().run_in_executor(None, _get)

    # ── Internal sync helpers ─────────────────────────────────────────────────

    def _search_track_sync(self, title: str, artist: str, ocr_raw: Optional[str] = None) -> Optional[TrackInfo]:
        """Synchronous Spotify search — runs in thread pool with cascading strategies."""
        if not self._client:
            logger.debug("Spotify client not available; skipping track search")
            return None

        clean_title = _clean_for_search(title)
        clean_artist = _clean_for_search(artist)

        strategies = [
            (f'track:"{clean_title}" artist:"{clean_artist}"', "exact"),            # 1. exact match
            (f'"{clean_title}" "{clean_artist}"', "exact"),                         # 2. strict quotes, no field
            (f'"{clean_title}" {clean_artist}', "fuzzy"),                           # 3. quote title only
            (f'{clean_title} {clean_artist}', "fuzzy"),                             # 4. loose text
            (clean_title, "fuzzy"),                                                 # 5. title only
        ]
        if ocr_raw:
            strategies.append((_clean_for_search(ocr_raw), "fuzzy"))               # 6. raw text

        auth_retried = False
        for query, confidence in strategies:
            if not query.strip():
                continue
                
            while True:
                try:
                    results = self._client.search(q=query, type="track", limit=1, market="US")
                    items = results.get("tracks", {}).get("items", [])
                    if items:
                        t = items[0]
                        
                        # Verification logic: if it was a title-only search, ensure some artist name overlap
                        if confidence == "fuzzy" and clean_artist:
                            found_artists = " ".join([a["name"].lower() for a in t["artists"]])
                            # Very basic fuzzy check — robust enough for now
                            overlap = sum(1 for w in clean_artist.lower().split() if w in found_artists)
                            if overlap == 0 and len(clean_artist.split()) > 0:
                                break  # Skip this result, try next strategy

                        return TrackInfo(
                            spotify_id=t["id"],
                            uri=t["uri"],
                            name=t["name"],
                            artist=t["artists"][0]["name"],
                            album=t["album"]["name"],
                            release_year=(t["album"].get("release_date") or "")[:4] or None,
                            duration_ms=t["duration_ms"],
                            popularity=t["popularity"],
                            preview_url=t.get("preview_url"),
                            spotify_url=t["external_urls"]["spotify"],
                            match_confidence=confidence,
                        )
                    break # success but no items, break out of while
                except SpotifyException as exc:
                    if exc.http_status == 401 and not auth_retried:
                        # Access token expired — rebuild the client with fresh credentials
                        logger.warning("Spotify 401 — refreshing client credentials")
                        self._client = _build_client_credentials_client()
                        auth_retried = True
                        continue   # retry this exact query
                    logger.warning(f"Spotify search error for '{query}': {exc}")
                    break
                except Exception as exc:
                    logger.warning(f"Spotify search error for '{query}': {exc}")
                    break
        
        return None

    def _create_playlist_sync(
        self,
        user_access_token: str,
        user_spotify_id: str,
        playlist_name: str,
        tracks: List[Dict],
    ) -> PlaylistCreationResult:
        client = _build_oauth_client(user_access_token)

        # Create empty playlist
        playlist = client.user_playlist_create(
            user=user_spotify_id,
            name=playlist_name,
            public=True,
            description="Auto-generated by YouTube Intelligent Classifier",
        )
        playlist_id = playlist["id"]
        playlist_url = playlist["external_urls"]["spotify"]
        logger.info(f"Created Spotify playlist: {playlist_name} ({playlist_id})")

        track_uris: List[str] = []
        not_found: List[str] = []

        for track in tracks:
            title = track.get("title", "")
            artist = track.get("artist", "")
            info = self._search_track_sync(title, artist)
            if info:
                track_uris.append(info.uri)
            else:
                not_found.append(f"{title} – {artist}")

        # Spotify allows max 100 tracks per add call
        CHUNK = 100
        for i in range(0, len(track_uris), CHUNK):
            chunk = track_uris[i : i + CHUNK]
            client.playlist_add_items(playlist_id, chunk)

        logger.info(
            f"Added {len(track_uris)} tracks to playlist "
            f"({len(not_found)} not found)"
        )
        return PlaylistCreationResult(
            playlist_id=playlist_id,
            playlist_url=playlist_url,
            name=playlist_name,
            tracks_added=len(track_uris),
            tracks_not_found=not_found,
        )
