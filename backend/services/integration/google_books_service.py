"""
services/integration/google_books_service.py

Google Books API integration.

Features
────────
- Book search by title (and optional author hint)
- Returns structured BookInfo with ISBN, author, description, purchase links
- Redis caching (24 hr TTL)
- Graceful degradation when API key is not configured
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote_plus

import httpx
from loguru import logger

from core.config import get_settings

settings = get_settings()

_BOOKS_BASE = "https://www.googleapis.com/books/v1/volumes"
_CACHE_TTL = 86_400  # 24 hours


@dataclass
class BookInfo:
    title: str
    authors: list[str]
    description: Optional[str]
    isbn: Optional[str]
    thumbnail: Optional[str]
    google_books_url: Optional[str]
    goodreads_url: str
    amazon_url: str
    published_date: Optional[str]
    page_count: Optional[int]
    found: bool = True


class GoogleBooksService:
    """Searches Google Books API and caches results in Redis."""

    def __init__(self):
        self._redis = None

    def _get_redis(self):
        if self._redis is None:
            try:
                import redis as redis_lib
                self._redis = redis_lib.from_url(settings.REDIS_URL, decode_responses=True)
            except Exception:
                self._redis = None
        return self._redis

    def _cache_key(self, query: str) -> str:
        h = hashlib.md5(query.lower().encode()).hexdigest()
        return f"google_books:{h}"

    async def search_book(self, title: str, author: Optional[str] = None) -> Optional[BookInfo]:
        """
        Search Google Books for a book by title (and optional author hint).
        Returns None if API key not configured or book not found.
        """
        if not settings.GOOGLE_BOOKS_API_KEY:
            logger.debug("GOOGLE_BOOKS_API_KEY not set — skipping Books enrichment")
            return None

        query = f"intitle:{title}"
        if author:
            query += f"+inauthor:{author}"

        cache_key = self._cache_key(query)
        redis = self._get_redis()

        # Check cache first
        if redis:
            try:
                cached = redis.get(cache_key)
                if cached:
                    data = json.loads(cached)
                    return BookInfo(**data) if data else None
            except Exception as exc:
                logger.debug(f"Redis cache miss for Books ({exc})")

        # Call Google Books API
        try:
            params = {
                "q": query,
                "maxResults": 1,
                "key": settings.GOOGLE_BOOKS_API_KEY,
                "langRestrict": "en",
            }
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(_BOOKS_BASE, params=params)
                resp.raise_for_status()
                data = resp.json()

            items = data.get("items") or []
            if not items:
                logger.info(f"[Books] No results for '{title}'")
                if redis:
                    redis.setex(cache_key, _CACHE_TTL, "null")
                return None

            vol = items[0].get("volumeInfo", {})
            isbn = None
            for identifier in vol.get("industryIdentifiers", []):
                if identifier.get("type") == "ISBN_13":
                    isbn = identifier["identifier"]
                    break

            book = BookInfo(
                title=vol.get("title", title),
                authors=vol.get("authors") or [],
                description=(vol.get("description") or "")[:400] or None,
                isbn=isbn,
                thumbnail=(vol.get("imageLinks") or {}).get("thumbnail"),
                google_books_url=vol.get("infoLink"),
                goodreads_url=f"https://www.goodreads.com/search?q={quote_plus(vol.get('title', title))}",
                amazon_url=f"https://www.amazon.com/s?k={quote_plus(vol.get('title', title))}",
                published_date=vol.get("publishedDate"),
                page_count=vol.get("pageCount"),
            )

            if redis:
                import dataclasses
                redis.setex(cache_key, _CACHE_TTL, json.dumps(dataclasses.asdict(book)))

            logger.info(f"[Books] Found: '{book.title}' by {', '.join(book.authors)}")
            return book

        except httpx.HTTPStatusError as exc:
            logger.warning(f"[Books] HTTP error for '{title}': {exc.response.status_code}")
        except Exception as exc:
            logger.error(f"[Books] Unexpected error for '{title}': {exc}")

        return None
