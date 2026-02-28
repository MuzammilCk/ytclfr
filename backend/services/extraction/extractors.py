"""
services/extraction/extractors.py

Category-specific information extraction pipelines.

Each extractor receives the transcript, metadata, and frame paths,
and returns a structured payload specific to that content type.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import spacy
from loguru import logger

# Lazy load spacy model
_nlp: Optional[Any] = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except Exception as exc:
            logger.warning(
                f"spaCy model en_core_web_sm unavailable ({exc}). Falling back to blank 'en' pipeline."
            )
            _nlp = spacy.blank("en")
    return _nlp


# ── Shared utilities ──────────────────────────────────────────────────────────

def _extract_named_entities(text: str) -> Dict[str, List[str]]:
    """Return NER results grouped by entity type."""
    nlp = _get_nlp()
    doc = nlp(text[:100_000])   # spacy limit
    entities: Dict[str, List[str]] = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, [])
        if ent.text not in entities[ent.label_]:
            entities[ent.label_].append(ent.text)
    return entities


def _extract_key_phrases(text: str, top_n: int = 15) -> List[str]:
    """Extract noun chunk phrases as key topics."""
    nlp = _get_nlp()
    doc = nlp(text[:50_000])
    phrase_freq: Dict[str, int] = {}

    if "parser" not in nlp.pipe_names:
        # Fallback when model data isn't available (e.g., restricted environments)
        tokens = [t.text.lower() for t in doc if t.is_alpha and len(t.text) > 3 and not t.is_stop]
        for tok in tokens:
            phrase_freq[tok] = phrase_freq.get(tok, 0) + 1
    else:
        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip().lower()
            if len(phrase) > 3:
                phrase_freq[phrase] = phrase_freq.get(phrase, 0) + 1

    return [p for p, _ in sorted(phrase_freq.items(), key=lambda x: -x[1])][:top_n]


# ── Ranklist extraction ───────────────────────────────────────────────────────

# Patterns like "1. The Dark Knight", "#1 - Avatar", "Number 10: Inception"
_RANK_PATTERNS = [
    re.compile(r"^(?:#|no\.?\s*|number\s*)?(\d{1,2})[\.\)\-:\s]+(.+)$", re.I | re.M),
    re.compile(r"^(\d{1,2})\s*[-\.]\s*(.+)$", re.I | re.M),
]


def _extract_ranked_list(text: str) -> List[Tuple[Optional[int], str]]:
    """
    Parse a ranked list from transcript/description text.
    Returns list of (rank, item_title) tuples, sorted by rank.
    """
    found: Dict[int, str] = {}
    for pattern in _RANK_PATTERNS:
        for m in pattern.finditer(text):
            try:
                rank = int(m.group(1))
                title = m.group(2).strip()
                # Skip if it looks like a time code or URL fragment
                if title and rank <= 100 and not title.startswith("http"):
                    found[rank] = title
            except (ValueError, IndexError):
                continue

    if not found:
        return []

    return sorted([(r, t) for r, t in found.items()], key=lambda x: x[0])


# ── Music extraction ──────────────────────────────────────────────────────────

_ARTIST_SPLIT = re.compile(r"\s*[-–—|×]\s*|\s+by\s+|\s+ft\.?\s+|\s+feat\.?\s+", re.I)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _parse_music_entries(text: str) -> List[Dict[str, Optional[str]]]:
    """
    Heuristically parse (title, artist) pairs from transcript.
    Looks for patterns: "Song Name - Artist" or numbered lists with hyphen separators.
    """
    tracks: List[Dict[str, Optional[str]]] = []
    lines = text.split("\n")
    rank = 0
    for line in lines:
        line = line.strip()
        if not line or len(line) < 4:
            continue
        # Try to split on artist separator
        parts = _ARTIST_SPLIT.split(line, maxsplit=1)
        if len(parts) == 2:
            title, artist = parts[0].strip(), parts[1].strip()
            year_match = _YEAR_RE.search(line)
            year = year_match.group() if year_match else None
            rank += 1
            tracks.append({
                "rank": rank,
                "title": title,
                "artist": artist,
                "year": year,
                "album": None,
                "genre": None,
            })
    return tracks


# ── Chapter detection ─────────────────────────────────────────────────────────

_TS_RE = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b")


def _parse_chapters_from_description(description: str) -> List[Dict[str, Any]]:
    """
    Extract YouTube chapters from description text.
    Common format: "0:00 Introduction\n1:30 Step 1: Setup"
    """
    chapters: List[Dict[str, Any]] = []
    for line in description.split("\n"):
        m = _TS_RE.search(line)
        if m:
            hrs = int(m.group(1)) if m.group(3) else 0
            mins = int(m.group(2)) if m.group(3) else int(m.group(1))
            secs = int(m.group(3)) if m.group(3) else int(m.group(2))
            start = hrs * 3600 + mins * 60 + secs
            title = line[m.end():].strip().lstrip("-:").strip()
            if title:
                chapters.append({"start_secs": start, "title": title})

    # Assign end times
    for i in range(len(chapters) - 1):
        chapters[i]["end_secs"] = chapters[i + 1]["start_secs"]
    if chapters:
        chapters[-1]["end_secs"] = None   # end of video

    return chapters


# ── Base extractor ────────────────────────────────────────────────────────────

class BaseExtractor(ABC):
    @abstractmethod
    def extract(
        self,
        transcript_text: str,
        segments: List[Dict],
        metadata: Dict[str, Any],
        frame_paths: List[str],
    ) -> Dict[str, Any]:
        ...


# ── Comedy extractor ──────────────────────────────────────────────────────────

class ComedyExtractor(BaseExtractor):
    def extract(self, transcript_text, segments, metadata, frame_paths):
        logger.info("Running ComedyExtractor")
        entities = _extract_named_entities(transcript_text)
        key_moments = self._detect_punchlines(segments)
        sentiment_arc = self._compute_sentiment_arc(segments)
        return {
            "type": "comedy",
            "key_moments": key_moments,
            "sentiment_arc": sentiment_arc,
            "named_entities": entities,
            "summary": f"Comedy video with {len(segments)} speech segments.",
        }

    @staticmethod
    def _detect_punchlines(segments: List[Dict]) -> List[Dict]:
        """Flag segments with common punchline / laugh indicators."""
        markers = ["haha", "hehe", "lol", "laugh", "got you", "got 'em"]
        punchlines = []
        for seg in segments:
            text_lower = seg.get("text", "").lower()
            if any(m in text_lower for m in markers):
                punchlines.append({
                    "type": "punchline",
                    "timestamp_secs": seg.get("start", 0),
                    "description": seg.get("text", "")[:80],
                })
        return punchlines

    @staticmethod
    def _compute_sentiment_arc(segments: List[Dict]) -> List[Dict]:
        """
        Proxy sentiment via avg_logprob (confidence) and no_speech_prob.
        In a production system, replace with a fine-tuned sentiment model.
        """
        arc = []
        for seg in segments[::3]:   # sample every 3rd segment
            score = 1.0 - seg.get("no_speech_prob", 0.5)
            arc.append({"time": seg.get("start", 0), "score": round(score, 3)})
        return arc


# ── Listicle extractor ────────────────────────────────────────────────────────

class ListicleExtractor(BaseExtractor):
    def extract(self, transcript_text, segments, metadata, frame_paths):
        logger.info("Running ListicleExtractor")
        combined = metadata.get("description", "") + "\n" + transcript_text
        ranked = _extract_ranked_list(combined)

        items = []
        for rank, title in ranked:
            # Clean common noise: "(2019)", "[HD]", etc.
            clean_title = re.sub(r"\(.*?\)|\[.*?\]", "", title).strip()
            items.append({
                "rank": rank,
                "title": clean_title,
                "description": None,    # will be enriched by TMDbService
                "year": None,
                "tmdb_rating": None,
                "tmdb_id": None,
                "poster_url": None,
                "streaming": None,
                "imdb_url": None,
            })

        return {
            "type": "listicle",
            "list_title": metadata.get("title", ""),
            "items": items,
            "total_count": len(items),
        }


# ── Music extractor ───────────────────────────────────────────────────────────

class MusicExtractor(BaseExtractor):
    def extract(self, transcript_text, segments, metadata, frame_paths):
        logger.info("Running MusicExtractor")
        # Try description first (richer formatting); fall back to transcript
        source = metadata.get("description", "") or transcript_text
        raw_tracks = _parse_music_entries(source)

        if len(raw_tracks) < 3:
            # Fall back to transcript
            raw_tracks = _parse_music_entries(transcript_text)

        tracks = []
        seen: set = set()
        for t in raw_tracks:
            key = (t.get("title", "") or "").strip().lower()
            if key and key in seen:
                continue
            seen.add(key)
            tracks.append({
                **t,
                "spotify": {
                    "spotify_url": None,
                    "spotify_id": None,
                    "found": False,
                },
                "timestamp_secs": None,
            })

        return {
            "type": "music",
            "tracks": tracks,
            "total_count": len(tracks),
            "spotify_playlist_url": None,   # set after Spotify call
        }


# ── Educational extractor ─────────────────────────────────────────────────────

class EducationalExtractor(BaseExtractor):
    def extract(self, transcript_text, segments, metadata, frame_paths):
        logger.info("Running EducationalExtractor")
        description = metadata.get("description", "")
        chapters = _parse_chapters_from_description(description)

        if not chapters:
            # Fall back to uniform chapter segmentation using segment groups
            chapters = self._auto_segment(segments)

        key_concepts = _extract_key_phrases(transcript_text)
        entities = _extract_named_entities(transcript_text)

        # Build chapter summaries from segment text in each time window
        enriched_chapters = []
        for i, ch in enumerate(chapters):
            start = ch["start_secs"]
            end = ch.get("end_secs") or float("inf")
            in_window = [
                s.get("text", "")
                for s in segments
                if start <= s.get("start", 0) < end
            ]
            summary = " ".join(in_window)[:300]
            enriched_chapters.append({
                "index": i + 1,
                "title": ch["title"],
                "start_secs": start,
                "end_secs": ch.get("end_secs"),
                "summary": summary,
                "key_concepts": _extract_key_phrases(summary, top_n=5),
                "screenshot_url": frame_paths[min(i * 5, len(frame_paths) - 1)] if frame_paths else None,
            })

        return {
            "type": "educational",
            "chapters": enriched_chapters,
            "key_concepts": key_concepts,
            "summary": transcript_text[:500],
            "named_entities": entities,
        }

    @staticmethod
    def _auto_segment(segments: List[Dict], n_chapters: int = 5) -> List[Dict]:
        """Divide segments into n_chapters equal time blocks if no chapter markers."""
        if not segments:
            return []
        total = segments[-1].get("end", segments[-1].get("start", 0))
        block = total / n_chapters
        return [
            {"start_secs": i * block, "end_secs": (i + 1) * block, "title": f"Part {i + 1}"}
            for i in range(n_chapters)
        ]


# ── Generic extractor ─────────────────────────────────────────────────────────

class GenericExtractor(BaseExtractor):
    def extract(self, transcript_text, segments, metadata, frame_paths):
        logger.info("Running GenericExtractor")
        entities = _extract_named_entities(transcript_text)
        key_points = _extract_key_phrases(transcript_text)
        return {
            "type": "generic",
            "summary": transcript_text[:600],
            "key_points": key_points,
            "named_entities": entities,
        }


# ── Shopping extractor ────────────────────────────────────────────────────────

class ShoppingExtractor(BaseExtractor):
    """
    Extracts shoppable products from a video using YOLO detections combined
    with NLP entity extraction from the title / description / transcript.

    YOLO detections are passed in via *frame_paths* after the pipeline has
    already run ``YOLODetector.detect()``.  The detections are stored on the
    instance by the pipeline before calling ``extract()``.
    """

    # COCO classes that are plausibly shoppable consumer products
    _PRODUCT_LABELS: frozenset = frozenset({
        "backpack", "umbrella", "handbag", "tie", "suitcase",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "orange", "pizza", "cake",
        "chair", "couch", "bed", "dining table", "toilet",
        "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "clock",
        "vase", "scissors", "hair drier", "toothbrush",
        "book", "teddy bear", "sports ball", "kite", "baseball bat",
        "skateboard", "surfboard", "tennis racket",
    })

    def __init__(self) -> None:
        self.detections: List[Any] = []   # set by pipeline before calling extract()

    def extract(
        self,
        transcript_text: str,
        segments: List[Dict],
        metadata: Dict[str, Any],
        frame_paths: List[str],
    ) -> Dict[str, Any]:
        logger.info("Running ShoppingExtractor")

        products = self._build_product_list()

        # Supplement with NLP-detected product-like PRODUCT / ORG entities
        entities = _extract_named_entities(transcript_text)
        brand_mentions = list({e for e in entities.get("ORG", []) if len(e) >= 3})[:10]
        product_mentions = list({e for e in entities.get("PRODUCT", []) if len(e) >= 3})[:10]

        # Merge NLP products that are not already covered by YOLO
        yolo_names = {p["name"].lower() for p in products}
        seen_nlp: set = set()
        for name in product_mentions:
            key = name.lower()
            if key not in yolo_names and key not in seen_nlp:
                seen_nlp.add(key)
                products.append({
                    "name": name,
                    "brand": None,
                    "category": "mentioned",
                    "frame_timestamps": [],
                    "detection_source": "nlp",
                    "confidence": None,
                    "search_url": self._google_shopping_url(name),
                    "prices_are_live": False,
                })

        return {
            "type": "shopping",
            "products": products,
            "brand_mentions": brand_mentions,
            "total_products": len(products),
            "summary": (
                f"Detected {len(products)} shoppable product(s) across the video. "
                f"Brand mentions: {', '.join(brand_mentions) or 'none'}."
            ),
        }

    def _build_product_list(self) -> List[Dict[str, Any]]:
        """Aggregate YOLO detections into a deduplicated product list."""
        label_map: Dict[str, Dict[str, Any]] = {}
        for det in self.detections:
            label: str = det.label
            if label not in self._PRODUCT_LABELS:
                continue
            if label not in label_map:
                label_map[label] = {
                    "name": label.replace("_", " ").title(),
                    "brand": None,
                    "category": self._infer_category(label),
                    "frame_timestamps": [],
                    "detection_source": "yolo",
                    "confidence": det.confidence,
                    "search_url": self._google_shopping_url(label),
                    "prices_are_live": False,   # disclaimer: prices are not fetched in real-time
                }
            frame_path = det.frame_path
            if frame_path not in label_map[label]["frame_timestamps"]:
                label_map[label]["frame_timestamps"].append(frame_path)
            if det.confidence > label_map[label]["confidence"]:
                label_map[label]["confidence"] = det.confidence

        return sorted(label_map.values(), key=lambda x: -x["confidence"])

    @staticmethod
    def _infer_category(label: str) -> str:
        """Map a COCO label to a broad product category."""
        electronics = {"tv", "laptop", "mouse", "keyboard", "cell phone", "remote"}
        clothing = {"backpack", "handbag", "tie", "umbrella", "suitcase"}
        kitchen = {
            "bottle", "cup", "fork", "knife", "spoon", "bowl",
            "microwave", "oven", "toaster", "refrigerator", "wine glass",
        }
        sports = {"sports ball", "kite", "baseball bat", "skateboard", "surfboard", "tennis racket"}
        home = {"chair", "couch", "bed", "dining table", "vase", "clock", "book", "teddy bear"}
        if label in electronics:
            return "Electronics"
        if label in clothing:
            return "Clothing & Accessories"
        if label in kitchen:
            return "Kitchen & Dining"
        if label in sports:
            return "Sports & Outdoors"
        if label in home:
            return "Home & Garden"
        return "General"

    @staticmethod
    def _google_shopping_url(query: str) -> str:
        """Return a Google Shopping URL for the given product query."""
        from urllib.parse import quote_plus
        return f"https://www.google.com/search?tbm=shop&q={quote_plus(query)}"


# ── Factory ───────────────────────────────────────────────────────────────────

_EXTRACTOR_MAP: Dict[str, BaseExtractor] = {
    "comedy": ComedyExtractor(),
    "listicle": ListicleExtractor(),
    "music": MusicExtractor(),
    "educational": EducationalExtractor(),
    "shopping": ShoppingExtractor(),
    "news": GenericExtractor(),
    "review": GenericExtractor(),
    "gaming": GenericExtractor(),
    "vlog": GenericExtractor(),
    "unknown": GenericExtractor(),
}


def get_extractor(category: str) -> BaseExtractor:
    return _EXTRACTOR_MAP.get(category, _EXTRACTOR_MAP["unknown"])
