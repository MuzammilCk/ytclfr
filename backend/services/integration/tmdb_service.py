"""
services/integration/tmdb_service.py

The Movie Database (TMDb) API integration.

Features
────────
- Movie and TV show search with year disambiguation
- Streaming availability via /watch/providers (powered by JustWatch data in TMDb)
- Poster and backdrop image URL construction
- Async via httpx
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx
from loguru import logger

from core.config import get_settings

settings = get_settings()


@dataclass
class StreamingAvailability:
    flatrate: List[str] = field(default_factory=list)   # subscription (Netflix, etc.)
    rent: List[str] = field(default_factory=list)
    buy: List[str] = field(default_factory=list)


@dataclass
class MovieInfo:
    tmdb_id: int
    title: str
    original_title: str
    year: Optional[str]
    vote_average: float
    vote_count: int
    overview: str
    genres: List[str]
    poster_url: Optional[str]
    backdrop_url: Optional[str]
    imdb_id: Optional[str]
    imdb_url: Optional[str]
    homepage: Optional[str]
    streaming: Optional[StreamingAvailability] = None


class TMDbService:
    """
    Async TMDb API client using httpx.

    All methods are async-native to avoid blocking the event loop.
    A single shared httpx.AsyncClient is reused across calls (connection pooling).
    """

    def __init__(self):
        self._api_key = settings.TMDB_API_KEY
        self._base = settings.TMDB_BASE_URL
        self._img_base = settings.TMDB_IMAGE_BASE
        self._client: Optional[httpx.AsyncClient] = None

    def is_available(self) -> bool:
        return bool(self._api_key)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base,
                timeout=10.0,
                headers={"Accept": "application/json"},
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── Public API ─────────────────────────────────────────────────────────────

    async def search_movie(
        self, title: str, year: Optional[str] = None
    ) -> Optional[MovieInfo]:
        """Search for a movie by title (and optionally release year)."""
        if not self._api_key:
            return None
        try:
            client = await self._get_client()
            params: Dict = {"api_key": self._api_key, "query": title, "include_adult": "false"}
            if year:
                params["year"] = year

            resp = await client.get("/search/movie", params=params)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if not results:
                return None

            # Pick the result with the most votes (proxy for relevance)
            best = max(results, key=lambda m: m.get("vote_count", 0))
            return await self._enrich_movie(best)
        except Exception as exc:
            logger.warning(f"TMDb search failed for '{title}': {exc}")
            return None

    async def search_tv_show(
        self, title: str, year: Optional[str] = None
    ) -> Optional[MovieInfo]:
        """Search for a TV show. Returns same MovieInfo structure for uniformity."""
        if not self._api_key:
            return None
        try:
            client = await self._get_client()
            params: Dict = {"api_key": self._api_key, "query": title}
            if year:
                params["first_air_date_year"] = year

            resp = await client.get("/search/tv", params=params)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if not results:
                return None
            best = max(results, key=lambda s: s.get("vote_count", 0))

            # Normalise TV fields to movie schema
            return MovieInfo(
                tmdb_id=best["id"],
                title=best.get("name", ""),
                original_title=best.get("original_name", ""),
                year=(best.get("first_air_date") or "")[:4] or None,
                vote_average=best.get("vote_average", 0.0),
                vote_count=best.get("vote_count", 0),
                overview=best.get("overview", ""),
                genres=[],   # not available in search results
                poster_url=self._poster(best.get("poster_path")),
                backdrop_url=self._backdrop(best.get("backdrop_path")),
                imdb_id=None,
                imdb_url=None,
                homepage=None,
            )
        except Exception as exc:
            logger.warning(f"TMDb TV search failed for '{title}': {exc}")
            return None

    async def get_streaming(
        self, tmdb_id: int, region: str = "US"
    ) -> Optional[StreamingAvailability]:
        """
        Fetch streaming availability from TMDb's /watch/providers endpoint.
        Data is sourced from JustWatch (via TMDb partnership).
        """
        if not self._api_key:
            return None
        try:
            client = await self._get_client()
            resp = await client.get(
                f"/movie/{tmdb_id}/watch/providers",
                params={"api_key": self._api_key},
            )
            resp.raise_for_status()
            data = resp.json().get("results", {})
            region_data = data.get(region, {})
            return StreamingAvailability(
                flatrate=[p["provider_name"] for p in region_data.get("flatrate", [])],
                rent=[p["provider_name"] for p in region_data.get("rent", [])],
                buy=[p["provider_name"] for p in region_data.get("buy", [])],
            )
        except Exception as exc:
            logger.warning(f"TMDb streaming lookup failed for {tmdb_id}: {exc}")
            return None

    async def enrich_list_items(
        self, items: List[Dict]
    ) -> List[Dict]:
        """
        Bulk-enrich a list of {title, year} dicts with TMDb data + streaming.
        Runs searches concurrently (up to 10 at a time) to stay within rate limits.
        """
        SEM = asyncio.Semaphore(10)

        async def enrich_one(item: Dict) -> Dict:
            async with SEM:
                info = await self.search_movie(
                    item.get("title", ""), item.get("year")
                )
                if info:
                    streaming = await self.get_streaming(info.tmdb_id)
                    info.streaming = streaming
                    item.update({
                        "tmdb_id": info.tmdb_id,
                        "tmdb_rating": round(info.vote_average, 1),
                        "poster_url": info.poster_url,
                        "description": info.overview[:200] if info.overview else None,
                        "imdb_url": info.imdb_url,
                        "streaming": {
                            "flatrate": info.streaming.flatrate if info.streaming else [],
                            "rent": info.streaming.rent if info.streaming else [],
                            "buy": info.streaming.buy if info.streaming else [],
                        } if info.streaming else None,
                    })
                return item

        return await asyncio.gather(*[enrich_one(item) for item in items])

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _enrich_movie(self, search_result: Dict) -> MovieInfo:
        """Fetch full movie details to get IMDB id and genres."""
        movie_id = search_result["id"]
        client = await self._get_client()
        try:
            resp = await client.get(
                f"/movie/{movie_id}",
                params={"api_key": self._api_key},
            )
            resp.raise_for_status()
            d = resp.json()
        except Exception:
            d = search_result

        imdb_id = d.get("imdb_id")
        return MovieInfo(
            tmdb_id=movie_id,
            title=d.get("title", ""),
            original_title=d.get("original_title", ""),
            year=(d.get("release_date") or "")[:4] or None,
            vote_average=d.get("vote_average", 0.0),
            vote_count=d.get("vote_count", 0),
            overview=d.get("overview", ""),
            genres=[g["name"] for g in d.get("genres", [])],
            poster_url=self._poster(d.get("poster_path")),
            backdrop_url=self._backdrop(d.get("backdrop_path")),
            imdb_id=imdb_id,
            imdb_url=f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else None,
            homepage=d.get("homepage"),
        )

    def _poster(self, path: Optional[str]) -> Optional[str]:
        return f"{self._img_base}{path}" if path else None

    def _backdrop(self, path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        return f"https://image.tmdb.org/t/p/w1280{path}"
