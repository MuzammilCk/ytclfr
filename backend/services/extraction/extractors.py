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

from services.vision.ocr_service import FrameOCRData

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
        frame_ocr_results: List[FrameOCRData],
    ) -> Dict[str, Any]:
        ...


# ── Comedy extractor ──────────────────────────────────────────────────────────

class ComedyExtractor(BaseExtractor):
    def extract(self, transcript_text, segments, metadata, frame_paths, frame_ocr_results):
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
    def extract(self, transcript_text, segments, metadata, frame_paths, frame_ocr_results):
        logger.info("Running ListicleExtractor")

        # ── Detect whether this is a book list ─────────────────────────────────
        is_book_list = self._is_book_list(metadata)

        items = self._parse_from_ocr(frame_ocr_results)

        if len(items) < 2:
            logger.info("Falling back to transcript/description for listicle extraction")
            combined = metadata.get("description", "") + "\n" + transcript_text
            ranked = _extract_ranked_list(combined)
            
            # If transcript has more items, use it
            if len(ranked) > len(items):
                items = []
                for rank, title in ranked:
                    clean_title = re.sub(r"\(.*?\)|\[.*?\]", "", title).strip()
                    items.append({
                        "rank": rank,
                        "title": clean_title,
                        "source": "transcript",
                        "timestamp_secs": None,
                        "source_frame_index": None,
                    })

        # Deduplicate the merged items in case there are duplicates
        unique_items = []
        seen = set()
        for item in items:
            key = (item.get("title", "") or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                if is_book_list:
                    # Book list enrichment fields (populated async by google_books_service)
                    item.update({
                        "item_type": "book",
                        "description": None,
                        "authors": None,
                        "isbn": None,
                        "thumbnail": None,
                        "google_books_url": None,
                        "goodreads_url": None,
                        "amazon_url": None,
                        "published_date": None,
                        "page_count": None,
                    })
                else:
                    # Movie/TV enrichment fields (populated async by TMDbService)
                    item.update({
                        "item_type": "media",
                        "description": None,
                        "year": None,
                        "tmdb_rating": None,
                        "tmdb_id": None,
                        "poster_url": None,
                        "streaming": None,
                        "imdb_url": None,
                    })
                unique_items.append(item)
        items = unique_items

        if len(items) < 2:
            tag_items = self._extract_from_tags(metadata.get("tags", []))
            hashtag_items = self._extract_from_description_hashtags(metadata.get("description", ""))

            combined_map = {}
            for item in tag_items + hashtag_items:
                key = item["title"].lower()
                if key not in combined_map:
                    combined_map[key] = item

            if combined_map:
                sorted_items = sorted(combined_map.values(), key=lambda x: x["confidence"], reverse=True)
                items = []
                for i, item in enumerate(sorted_items):
                    base = {
                        "rank": i + 1,
                        "title": item["title"],
                        "source": item["source"],
                        "timestamp_secs": None,
                        "source_frame_index": None,
                    }
                    if is_book_list:
                        base.update({
                            "item_type": "book",
                            "description": None,
                            "authors": None,
                            "isbn": None,
                            "thumbnail": None,
                            "google_books_url": None,
                            "goodreads_url": None,
                            "amazon_url": None,
                            "published_date": None,
                            "page_count": None,
                        })
                    else:
                        base.update({
                            "item_type": "media",
                            "description": None,
                            "year": None,
                            "tmdb_rating": None,
                            "tmdb_id": None,
                            "poster_url": None,
                            "streaming": None,
                            "imdb_url": None,
                        })
                    items.append(base)

        summary = None
        if items and not transcript_text.strip().replace(".", ""):
            names = ", ".join(i["title"] for i in items[:5])
            suffix = f" and {len(items) - 5} more" if len(items) > 5 else ""
            summary = f"A {'book' if is_book_list else 'media'} list featuring: {names}{suffix}."

        return {
            "type": "listicle",
            "list_title": metadata.get("title", ""),
            "summary": summary,
            "is_book_list": is_book_list,
            "items": items,
            "total_count": len(items),
        }

    @staticmethod
    def _clean_listicle_title(raw_text: str) -> Tuple[str, Optional[int]]:
        """
        Extract title and optional year from dirty OCR text.
        e.g., "#1 THE SHAWSHANK REDEMPTION (1994)" -> ("The Shawshank Redemption", 1994)
        """
        # 1. Strip rank prefix if any
        rank_strip_re = re.compile(r"^(?:top\s+|best\s+)?(?:#|no\.?\s*|number\s*)?\d{1,3}[\.\)\-:\s]+", re.I)
        text = rank_strip_re.sub("", raw_text).strip()
        
        # 2. Extract year (4 digits in parens or brackets)
        year = None
        year_re = re.compile(r"[\(\[](19[0-9]{2}|20[0-2][0-9])[\)\]]")
        year_match = year_re.search(text)
        if year_match:
            year = int(year_match.group(1))
            text = text[:year_match.start()] + text[year_match.end():]
        
        # 3. Strip noise words / phrases
        noise_phrases = [
            "THE BEST", "BEST MOVIE EVER", "GREATEST EVER", "MUST WATCH",
            "TOP MOVIE", "NUMBER ONE", "WATCH", "MOVIE", "FILM"
        ]
        text_upper = text.upper()
        for phrase in noise_phrases:
            if text_upper.startswith(phrase):
                # Cut it off
                text = text[len(phrase):].strip(" -:")
                text_upper = text.upper()
        
        # 4. Clean up and title case
        clean_title = re.sub(r"\(.*?\)|\[.*?\]", "", text).strip()
        clean_title = re.sub(r"\s+", " ", clean_title) # reduce multiple spaces
        return clean_title.title(), year

    def _parse_from_ocr(self, ocr_results: List[FrameOCRData]) -> List[Dict[str, Any]]:
        items = []
        # Support Numbered: 1., #1, No. 1, Number 1, 1), TOP 1, BEST #1
        rank_re = re.compile(r"^(?:top\s+|best\s+)?(?:#|no\.?\s*|number\s*)?(\d{1,3})[\.\)\-:\s]+", re.I)
        
        for frame in ocr_results:
            if not frame.has_content:
                continue
                
            lines = [line.strip() for line in frame.cleaned_text.split('\n') if line.strip()]
            for i, line in enumerate(lines):
                if len(line) < 2:
                    continue
                    
                rank = None
                title = None
                rank_match = rank_re.match(line)
                if rank_match:
                    rank = int(rank_match.group(1))
                    title = line[rank_match.end():].strip()
                    if not title and i + 1 < len(lines):
                        title = lines[i+1].strip()
                elif line.isdigit() and i + 1 < len(lines):
                    # Multi-line
                    rank = int(line)
                    title = lines[i+1].strip()
                    
                if title and rank is not None and rank <= 100:
                    clean_title, year_hint = self._clean_listicle_title(title)
                    if len(clean_title) >= 2:
                        items.append({
                            "rank": rank,
                            "title": clean_title,
                            "year": year_hint,
                            "source": "ocr",
                            "timestamp_secs": frame.timestamp_secs,
                            "source_frame_index": frame.frame_index,
                        })
                        
        return sorted(items, key=lambda x: x.get("timestamp_secs", float('inf')))

    @staticmethod
    def _extract_from_tags(tags: List[str]) -> List[Dict[str, Any]]:
        noise_words = {
            "shorts", "youtube", "viral", "trending", "subscribe", "like", "follow", "edit", "video", "clip", "scene", "part",
            "best", "top", "only", "all", "time", "ever", "must", "watch", "movie", "movies", "film", "films", "show", "shows",
            "list", "ranking", "ranked", "review", "reaction", "cinema", "hollywood", "netflix", "hbo", "disney", "slowed",
            "reverb", "lofi", "cover", "remix", "music", "song", "songs", "track", "playlist", "album"
        }
        items = []
        for i, tag in enumerate(tags):
            tag = tag.lower().strip()
            if tag.startswith("#"):
                tag = tag[1:]
            
            for s in ["edit", "slowed", "reverb", "remix", "clip", "scene"]:
                if tag.endswith(s):
                    tag = tag[:-len(s)].strip()
            
            if len(tag) < 3 or tag in noise_words or tag.startswith("http") or tag.startswith("@"):
                continue
                
            items.append({
                "title": tag.title(),
                "source": "tag",
                "confidence": 1.0 - (i * 0.01)
            })
        return items

    @staticmethod
    def _extract_from_description_hashtags(description: str) -> List[Dict[str, Any]]:
        noise_words = {
            "shorts", "youtube", "viral", "trending", "subscribe", "like", "follow", "edit", "video", "clip", "scene", "part",
            "best", "top", "only", "all", "time", "ever", "must", "watch", "movie", "movies", "film", "films", "show", "shows",
            "list", "ranking", "ranked", "review", "reaction", "cinema", "hollywood", "netflix", "hbo", "disney", "slowed",
            "reverb", "lofi", "cover", "remix", "music", "song", "songs", "track", "playlist", "album"
        }
        items = []
        hashtags = re.findall(r"#(\w+)", description)
        for i, tag in enumerate(hashtags):
            tag = tag.strip()
            split_tag = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", tag)
            
            tag_clean = split_tag.lower()
            for s in ["edit", "slowed", "reverb", "remix", "clip", "scene"]:
                if tag_clean.endswith(s):
                    tag_clean = tag_clean[:-len(s)].strip()
                    split_tag = split_tag[:-len(s)].strip()

            if len(tag_clean) < 3 or tag_clean in noise_words:
                continue
                
            items.append({
                "title": split_tag.title(),
                "source": "description_hashtag",
                "confidence": 0.9 - (i * 0.01)
            })
        return items

    @staticmethod
    def _is_book_list(metadata: Dict[str, Any]) -> bool:
        """
        Detect whether this listicle is a book/reading list.
        Uses title, description, and tags heuristics.
        """
        _BOOK_SIGNALS = re.compile(
            r"\b(books?|novels?|reads?|reading list|must[- ]read|book recommendations?|"
            r"bibliography|goodreads|bestsellers?|authors?)\b",
            re.I,
        )
        combined = " ".join([
            metadata.get("title", ""),
            metadata.get("description", "")[:500],
            " ".join(metadata.get("tags", [])),
        ])
        return bool(_BOOK_SIGNALS.search(combined))


# ── Music extractor ───────────────────────────────────────────────────────────

class MusicExtractor(BaseExtractor):
    def extract(self, transcript_text, segments, metadata, frame_paths, frame_ocr_results):
        logger.info("Running MusicExtractor")
        
        raw_tracks = self._parse_from_ocr(frame_ocr_results)
        
        if len(raw_tracks) < 3:
            logger.info(f"Only found {len(raw_tracks)} tracks via OCR. Falling back to transcript/description.")
            source = metadata.get("description", "") or transcript_text
            fallback_tracks = _parse_music_entries(source)
            if len(fallback_tracks) < 3:
                fallback_tracks = _parse_music_entries(transcript_text)
                
            if len(fallback_tracks) > len(raw_tracks):
                raw_tracks = fallback_tracks
                # Inject None for missing properties
                for t in raw_tracks:
                    t["timestamp_secs"] = None
                    t["source_frame_index"] = None
                    t["source_frame_path"] = None

        tracks = []
        seen: set = set()
        
        # Deduplication: rely on chronological sort from frame OCR or source transcript
        for t in raw_tracks:
            key = (t.get("title", "") or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            tracks.append({
                **t,
                "spotify": {
                    "spotify_url": None,
                    "spotify_id": None,
                    "found": False,
                },
            })

        return {
            "type": "music",
            "tracks": tracks,
            "total_count": len(tracks),
            "spotify_playlist_url": None,
        }

    def _parse_from_ocr(self, ocr_results: List[FrameOCRData]) -> List[Dict[str, Any]]:
        tracks = []
        artist_split = re.compile(r"\s*[-–—|×]\s*|\s+by\s+|\s+ft\.?\s+|\s+feat\.?\s+", re.I)
        year_re = re.compile(r"\b(19|20)\d{2}\b")
        rank_re = re.compile(r"^(?:#|no\.?\s*|number\s*)?(\d{1,3})[\.\)\-:\s]+", re.I)
        
        for frame in ocr_results:
            if not frame.has_content:
                continue
                
            lines = [line.strip() for line in frame.cleaned_text.split('\n') if line.strip()]
            for i, line in enumerate(lines):
                if len(line) < 4:
                    continue
                    
                rank = None
                rank_match = rank_re.match(line)
                if rank_match:
                    rank = int(rank_match.group(1))
                    line = line[rank_match.end():].strip()
                elif line.isdigit() and i + 1 < len(lines):
                    # Multiline: rank is on this line, next line is title/artist
                    rank = int(line)
                    line = lines[i+1].strip()
                    
                parts = artist_split.split(line, maxsplit=1)
                
                title, artist = None, None
                if len(parts) == 2:
                    title, artist = parts[0].strip(), parts[1].strip()
                else:
                    # Check Artist "Song" or Song (Artist)
                    quotes_match = re.search(r'^(.*?)\s*["“”](.*?)["“”]', line)
                    if quotes_match:
                        artist, title = quotes_match.group(1).strip(), quotes_match.group(2).strip()
                    else:
                        parens_match = re.search(r'^(.*?)\s*\((.*?)\)', line)
                        if parens_match:
                            title, artist = parens_match.group(1).strip(), parens_match.group(2).strip()
                            
                if title and artist:
                    title = title.strip('"“”\'')
                    year_match = year_re.search(line)
                    
                    tracks.append({
                        "rank": rank,
                        "title": title,
                        "artist": artist,
                        "year": year_match.group() if year_match else None,
                        "timestamp_secs": frame.timestamp_secs,
                        "source_frame_index": frame.frame_index,
                        "source_frame_path": frame.frame_path,
                        "album": None,
                        "genre": None,
                    })
                    
        # Return sorted by timestamp so earliest is processed first for deduplication
        return sorted(tracks, key=lambda x: x.get("timestamp_secs", float('inf')))


# ── Educational extractor ─────────────────────────────────────────────────────

class EducationalExtractor(BaseExtractor):
    def extract(self, transcript_text, segments, metadata, frame_paths, frame_ocr_results):
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
    def extract(self, transcript_text, segments, metadata, frame_paths, frame_ocr_results):
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
        frame_ocr_results: List[FrameOCRData],
    ) -> Dict[str, Any]:
        logger.info("Running ShoppingExtractor")

        products = self._build_product_list()

        # Build OCR map for quick frame lookup
        ocr_map = {f.frame_path: f for f in frame_ocr_results if f.has_content}
        
        # Merge OCR context onto YOLO products
        price_re = re.compile(r"\$\d+(?:,\d{3})*(?:\.\d{2})?")
        
        for p in products:
            best_name = None
            best_price = None
            
            for f_path in p.get("frame_timestamps", []):
                if f_path in ocr_map:
                    text = ocr_map[f_path].cleaned_text
                    lines = text.split('\n')
                    for line in lines:
                        # Find price
                        price_match = price_re.search(line)
                        price_str = price_match.group(0) if price_match else None
                        
                        # Find potential name (avoiding just the price string itself)
                        name_str = re.sub(r"\$\d+(?:,\d{3})*(?:\.\d{2})?", "", line)
                        name_str = re.sub(r"Product Name|Price", "", name_str, flags=re.I).strip(' -|:')
                        
                        if len(name_str) > 3 and not best_name:
                            best_name = name_str
                            
                        if price_str and not best_price:
                            best_price = price_str
                            
            if best_name:
                p["name"] = best_name
                p["detection_source"] = "yolo+ocr"
                p["search_url"] = self._google_shopping_url(best_name)
            if best_price:
                p["price"] = best_price

        # Supplement with NLP-detected product-like PRODUCT / ORG entities
        entities = _extract_named_entities(transcript_text)
        brand_mentions = list({e for e in entities.get("ORG", []) if len(e) >= 3})[:10]
        product_mentions = list({e for e in entities.get("PRODUCT", []) if len(e) >= 3})[:10]

        # Merge NLP products that are not already covered by YOLO/OCR
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
                    "price": None,
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
                    "price": None,
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


# ── Recipe extractor ──────────────────────────────────────────────────────────

class RecipeExtractor(BaseExtractor):
    def extract(self, transcript_text, segments, metadata, frame_paths, frame_ocr_results):
        logger.info("Running RecipeExtractor")
        
        ingredients = self._extract_ingredients(frame_ocr_results)
        steps = self._extract_steps(transcript_text, frame_ocr_results)
        
        # Look for prep/cook time in description
        desc = metadata.get("description", "").lower()
        prep_time = self._extract_time(desc, "prep")
        cook_time = self._extract_time(desc, "cook")
        
        # Look for servings
        servings_match = re.search(r"(?:yields|servings|serves)\s*:\s*(\d+)", desc)
        servings = int(servings_match.group(1)) if servings_match else None
        
        return {
            "type": "recipe",
            "title": metadata.get("title", ""),
            "ingredients": ingredients,
            "steps": steps,
            "servings": servings,
            "prep_time": prep_time,
            "cook_time": cook_time,
        }

    def _extract_ingredients(self, ocr_results: List[FrameOCRData]) -> List[Dict]:
        ingredients = []
        seen = set()
        
        # Unit normalization map
        unit_map = {
            "cup": "cup", "cups": "cup", "c": "cup",
            "tsp": "tsp", "teaspoon": "tsp", "teaspoons": "tsp",
            "tbsp": "tbsp", "tablespoon": "tbsp", "tablespoons": "tbsp",
            "oz": "oz", "ounce": "oz", "ounces": "oz",
            "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb",
            "g": "g", "gram": "g", "grams": "g",
            "kg": "kg", "kilogram": "kg", "kilograms": "kg",
            "ml": "ml", "milliliter": "ml", "milliliters": "ml",
            "l": "l", "liter": "l", "liters": "l",
            "pinch": "pinch", "pinches": "pinch",
            "dash": "dash", "dashes": "dash",
            "clove": "clove", "cloves": "clove",
            "piece": "piece", "pieces": "piece",
            "slice": "slice", "slices": "slice",
            "can": "can", "cans": "can",
        }

        # Match fractions "1/2", mixed "1 1/2", decimals "1.5", plain "2"
        qty_re = r"(\d+(?:\s+\d+)?/\d+|\d+\.\d+|\d+)"
        # Units
        units_pattern = r"(?:" + "|".join(unit_map.keys()) + r")"
        
        # Pattern 1: Qty Unit Name (e.g., 2 cups flour)
        ing_pattern = re.compile(rf"^{qty_re}\s+{units_pattern}\b\s+(.+)$", re.I)
        # Pattern 2: Qty Name (e.g., 3 eggs, 1/2 onion) -- no unit
        ing_pattern_nounit = re.compile(rf"^{qty_re}\s+(.+)$", re.I)
        # Pattern 3: Unit Name (e.g., A pinch of salt, pinch of pepper)
        ing_pattern_text_qty = re.compile(rf"^(?:a\s+)?({units_pattern})\s+(?:of\s+)?(.+)$", re.I)

        def parse_qty(q_str: str) -> Optional[float]:
            if not q_str: return None
            q_str = q_str.strip()
            try:
                if "/" in q_str:
                    parts = q_str.split()
                    if len(parts) == 2:
                        num, den = parts[1].split("/")
                        return float(parts[0]) + (float(num) / float(den))
                    else:
                        num, den = q_str.split("/")
                        return float(num) / float(den)
                return float(q_str)
            except ValueError:
                return None

        for frame in ocr_results:
            if not frame.has_content:
                continue
                
            lines = [line.strip() for line in frame.cleaned_text.split('\n') if line.strip()]
            for line in lines:
                # Strip leading list bullets from OCR
                line = re.sub(r"^[-*•]\s+", "", line)

                qty = None
                unit = None
                name = None

                m1 = ing_pattern.match(line)
                if m1:
                    qty = parse_qty(m1.group(1))
                    match_unit = re.search(units_pattern, line[m1.end(1):], re.I)
                    if match_unit:
                        unit = unit_map.get(match_unit.group(0).lower())
                    name = m1.group(2).strip()
                else:
                    m2 = ing_pattern_nounit.match(line)
                    if m2:
                        qty = parse_qty(m2.group(1))
                        name = m2.group(2).strip()
                    else:
                        m3 = ing_pattern_text_qty.match(line)
                        if m3:
                            qty = None
                            unit = unit_map.get(m3.group(1).lower())
                            name = m3.group(2).strip()

                if name:
                    # Filter junk like just dots
                    name = re.sub(r'^[-\.\s]+', '', name).strip()
                    if len(name) < 2:
                        continue
                        
                    key = name.lower()
                    if key not in seen:
                        seen.add(key)
                        ingredients.append({
                            "quantity": qty,
                            "unit": unit,
                            "name": name,
                        })
                            
        return ingredients

    def _extract_steps(self, transcript: str, ocr_results: List[FrameOCRData]) -> List[Dict]:
        steps = []
        step_idx = 1
        
        step_re = re.compile(r"^step\s+(\d+)[\.:\s]+(.+)", re.I)
        for frame in ocr_results:
             if not frame.has_content:
                continue
             lines = [line.strip() for line in frame.cleaned_text.split('\n') if line.strip()]
             for line in lines:
                 m = step_re.match(line)
                 if m:
                     steps.append({
                         "index": step_idx,
                         "text": m.group(2).strip(),
                         "timestamp_secs": frame.timestamp_secs
                     })
                     step_idx += 1
                     
        if not steps and transcript:
            sentences = [s.strip() for s in transcript.split('.') if s.strip()]
            keywords = ["first", "second", "next", "then", "now", "finally", "add", "mix", "stir", "bake", "cook", "pour"]
            for s in sentences:
                lower = s.lower()
                if any(lower.startswith(k) for k in keywords) and len(s) > 15:
                    steps.append({
                        "index": step_idx,
                        "text": s,
                        "timestamp_secs": None
                    })
                    step_idx += 1
                    
        return steps

    def _extract_time(self, text: str, prefix: str) -> Optional[str]:
        m = re.search(rf"{prefix}\s*(?:time)?\s*:\s*(\d+\s*(?:min|mins|minutes|hr|hrs|hours))", text)
        return m.group(1) if m else None


# ── Factory ───────────────────────────────────────────────────────────────────

_EXTRACTOR_MAP: Dict[str, BaseExtractor] = {
    "comedy": ComedyExtractor(),
    "listicle": ListicleExtractor(),
    "music": MusicExtractor(),
    "educational": EducationalExtractor(),
    "news": GenericExtractor(),
    "review": GenericExtractor(),
    "gaming": GenericExtractor(),
    "recipe": RecipeExtractor(),
    "vlog": GenericExtractor(),
    "unknown": GenericExtractor(),
}


def get_extractor(category: str) -> BaseExtractor:
    if category == "shopping":
        return ShoppingExtractor()
    return _EXTRACTOR_MAP.get(category, _EXTRACTOR_MAP["unknown"])
