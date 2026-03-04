# YTCLFR — Build Specification
## Technical Architecture & Phased Implementation Plan
### Version 1.0 | Living Document — Update After Every Phase

---

## DOCUMENT PURPOSE

This document is the single source of truth for building YTCLFR. Every phase must be fully implemented, tested, and verified before the next phase begins. This document must be read in full before starting any implementation. It must be updated with completion status, notes, and any architectural decisions made during each phase.

**Read `context.md` before reading this document. Read this document before writing a single line of code.**

---

## CURRENT STATE BASELINE

The existing codebase has the following components already built. These are the foundation. Do not delete them — refactor and extend them:

```
backend/
  main.py                          — FastAPI app, middleware, health check ✅
  core/config.py                   — Settings, env vars, validation ✅
  db/database.py                   — Postgres async + sync, MongoDB, Redis ✅
  db/models.py                     — SQLAlchemy ORM models ✅
  models/schemas.py                — Pydantic schemas ✅
  api/routes/analysis.py           — Analysis submission + result retrieval ✅
  api/routes/analytics.py          — Usage analytics ✅
  api/routes/auth.py               — JWT auth, Spotify OAuth ✅
  api/routes/users.py              — Admin user management ✅
  api/middleware/rate_limiter.py   — Redis sliding window rate limiter ✅
  services/pipeline.py             — Celery task orchestrator ⚠️ NEEDS REWRITE
  services/video_processor/downloader.py   — yt-dlp download ✅
  services/video_processor/frame_extractor.py — OpenCV frame sampling ✅
  services/audio_processor/transcriber.py  — faster-whisper ⚠️ NEEDS EXTENSION
  services/classification/classifier.py    — Multi-modal classifier ⚠️ NEEDS FIX
  services/extraction/extractors.py        — Category extractors ⚠️ NEEDS REWRITE
  services/extraction/llm_extractor.py     — Phi-3-mini llama.cpp ✅
  services/integration/spotify_service.py  — Spotify Web API ✅
  services/integration/tmdb_service.py     — TMDb API ✅
  services/vision/ocr_service.py           — Tesseract OCR ⚠️ NEEDS REWRITE
  services/vision/yolo_detector.py         — YOLOv8 object detection ✅
frontend/
  src/App.jsx                       — Full React UI ⚠️ NEEDS NEW COMPONENTS
```

**Core Problems to Fix (identified before phased build):**
1. OCR runs on only 8–20 frames. Must run on ALL frames.
2. OCR runs AFTER extraction. Must run BEFORE classification.
3. Extractors parse audio transcript, not OCR. Must be inverted.
4. Per-frame data is never stored in MongoDB. Must be added.
5. No translation support. Whisper `task="translate"` not used.
6. Frame files are never served to the frontend. No static file route.
7. Classification weights missing — `checkpoints/` is empty.

---

## ARCHITECTURE OVERVIEW

### Data Flow (Correct Architecture)

```
YouTube URL
    ↓
[Step 1] Download
    ├── Video file (.mp4)
    ├── Audio file (.wav, 16kHz mono)
    └── Metadata (title, tags, description, thumbnail)
    ↓
[Step 2] Frame Extraction
    └── N frames as JPEGs with timestamps
    ↓
[Step 3] Full-Frame OCR  ← PRIMARY DATA SOURCE
    └── {frame_index, timestamp_secs, ocr_text, frame_path}[] per frame
    ↓
[Step 4] Audio Transcription  ← SUPPLEMENTARY
    ├── transcript_original (native language)
    └── transcript_english (Whisper translate if non-English)
    ↓
[Step 5] Classification
    Input: OCR text (all frames) + title + tags + transcript
    Output: category + confidence + all_scores
    ↓
[Step 6] Category-Specific Extraction
    Input: frame_ocr_results[] + transcript_english + metadata
    Output: structured items (songs / movies / products / steps)
    ↓
[Step 7] External Enrichment
    Music   → Spotify search → playlist creation
    Listicle → TMDb search → ratings + streaming
    Shopping → Google Shopping URL construction
    Recipe  → ingredient normalization
    ↓
[Step 8] Persist to MongoDB
    Full result including per-frame data
    ↓
[Step 9] Update Postgres
    Analysis status → complete
    Video category + confidence written back
    ↓
Result returned to frontend
```

### Technology Stack (No Changes to Core Stack)

| Layer | Technology | Version |
|---|---|---|
| API | FastAPI | 0.115.x |
| Task Queue | Celery + Redis | 5.x |
| Primary DB | PostgreSQL 16 | asyncpg |
| Document Store | MongoDB 7 | Motor (async) |
| Cache | Redis 7 | redis-py async |
| ML — Transcription | faster-whisper | latest |
| ML — OCR | Tesseract 4 + pytesseract | latest |
| ML — Vision | YOLOv8n (ultralytics) | latest |
| ML — Classification | EfficientNet-B0 + BERT | torchvision / transformers |
| ML — LLM Extraction | Phi-3-mini GGUF (llama.cpp) | latest |
| Music API | Spotify Web API (spotipy) | latest |
| Movie API | TMDb API (httpx) | latest |
| Frontend | React 18 + Vite | latest |
| Container | Docker + docker-compose | latest |

---

## PHASE 0 — FOUNDATION HARDENING
**Goal:** Fix all known bugs in the existing codebase before building new features.
**Duration:** 1–2 days
**Status:** [x] COMPLETE

### 0.1 — Fix OCR Service: Run on All Frames

**File:** `backend/services/vision/ocr_service.py`

**Current problem:** `max_frames=8` in pre-classification call. `max_frames=20` in full extraction. For a 10-minute music countdown, 60 frames are extracted but only 20 are OCR'd. Songs visible only in the other 40 frames are missed forever.

**Required change:**
- Remove the `max_frames` parameter entirely from the primary extraction path
- `extract_from_frames()` must accept ALL frame paths and process every single one
- Keep a `max_frames` parameter only for the lightweight pre-classification call (keep at 8)
- Each `OCRResult` must include: `frame_index`, `timestamp_secs`, `frame_path`, `raw_text`, `cleaned_text`, `confidence`
- Add parallel processing: use `asyncio.gather` with a semaphore of 8 to prevent memory exhaustion

**Output schema per frame:**
```python
@dataclass
class OCRResult:
    frame_index: int
    frame_path: str
    timestamp_secs: float
    raw_text: str
    cleaned_text: str
    confidence: float  # 0–100 mean Tesseract word confidence
    has_content: bool  # True if cleaned_text has >= 3 meaningful words
```

### 0.2 — Fix Pipeline: OCR Before Classification

**File:** `backend/services/pipeline.py`

**Current order:**
```
download → frames → transcribe → classify → extract → (sometimes OCR)
```

**Required order:**
```
download → frames → OCR (all frames) → transcribe → classify → extract
```

**Required change:** Move full OCR to immediately after `frame_extractor.extract()`. Pass `ocr_results` (the full list of per-frame OCR objects) into classification and extraction. Remove all OCR calls from inside extractor methods — OCR is done once, at the pipeline level.

### 0.3 — Fix Training Script Category Mismatch

**File:** `scripts/train_frame_classifier.py` and `scripts/train_text_classifier.py`

**Problem:** Both scripts define 8 categories. `classifier.py` has 9 (includes "shopping"). Model head shape mismatch causes crash on load.

**Fix:** Add `"shopping"` to `CATEGORIES` list. Change `N_CLASSES = 9`. Verify both files match `classifier.py` exactly.

### 0.4 — Fix DownloadResult Missing temp_dir

**File:** `backend/services/video_processor/downloader.py`

**Problem:** `pipeline.py` checks `hasattr(download_result, "temp_dir")` but the dataclass has no such field. Temp dirs are never cleaned up.

**Fix:** Add `temp_dir: Optional[str] = None` to `DownloadResult` dataclass.

### 0.5 — Fix asyncio.new_event_loop() antipattern

**File:** `backend/services/pipeline.py`

**Problem:** `loop = asyncio.new_event_loop()` + `asyncio.set_event_loop(loop)` is deprecated and causes issues with Python 3.12+.

**Fix:** Replace with `asyncio.run()` or properly structured async/sync bridge using `concurrent.futures`.

### 0.6 — Fix list endpoint: include category

**File:** `backend/api/routes/analysis.py`

**Problem:** `GET /api/v1/analyses/` already includes category (fixed in current code with `selectinload(Analysis.video)`). Verify this is working. The frontend `HistoryPanel` should show category correctly.

### Completion Criteria for Phase 0:
- [x] OCR runs on every frame with no frame count limit
- [x] OCR runs before transcription in pipeline
- [x] Training scripts have 9 categories matching classifier.py
- [x] DownloadResult has temp_dir field
- [x] No asyncio.new_event_loop() in pipeline
- [x] All existing tests pass: `pytest tests/ -v`

---

## PHASE 1 — FRAME-FIRST EXTRACTION ENGINE
**Goal:** Completely rewrite extractors to use per-frame OCR as primary source.
**Duration:** 3–4 days
**Status:** [x] COMPLETE
**Prerequisite:** Phase 0 complete and all tests passing.

### 1.1 — Per-Frame OCR Data Structure

**File:** `backend/services/vision/ocr_service.py`

Add a new return type for the full OCR pass:

```python
@dataclass
class FrameOCRData:
    frame_index: int
    frame_path: str
    timestamp_secs: float
    raw_text: str
    cleaned_text: str
    confidence: float
    has_content: bool
    # Detected patterns (populated by the extraction layer, not OCR itself)
    detected_items: List[Dict] = field(default_factory=list)

@dataclass  
class VideoOCRResult:
    frames: List[FrameOCRData]
    aggregated_text: str          # All frame text deduplicated and joined
    content_frames: List[FrameOCRData]  # Only frames with has_content=True
    total_frames_processed: int
    frames_with_text: int
```

### 1.2 — Rewrite MusicExtractor

**File:** `backend/services/extraction/extractors.py`

**Current behavior:** Parses `metadata["description"]` or `transcript_text` for "Song - Artist" patterns.

**Required behavior:**
1. Primary source: iterate through `frame_ocr_results` (list of `FrameOCRData`)
2. For each frame with `has_content=True`, run music pattern matching
3. Pattern library must handle all real-world formats found in YouTube music countdowns:
   - `#1 Song Name - Artist Name`
   - `1. Song Name by Artist Name`
   - `Song Name | Artist Name`
   - `Song Name — Artist Name` (em dash)
   - `Artist Name "Song Name"`
   - `Song Name (Artist Name)`
   - `#20 "Song Name" - Artist Name (Year)`
   - Lines where rank appears on one line and title on next
4. Deduplication: same song found in multiple frames → keep once, use earliest timestamp
5. Fallback: if fewer than 3 songs found via OCR, fall back to description/transcript parsing
6. Each track includes: `rank`, `title`, `artist`, `timestamp_secs`, `source_frame_index`, `source_frame_path`

**Method signature:**
```python
def extract(
    self,
    transcript_text: str,
    segments: List[Dict],
    metadata: Dict[str, Any],
    frame_paths: List[str],
    frame_ocr_results: List[FrameOCRData],  # NEW — primary source
) -> Dict[str, Any]:
```

### 1.3 — Rewrite ListicleExtractor

**File:** `backend/services/extraction/extractors.py`

**Required behavior:**
1. Primary source: `frame_ocr_results`
2. Pattern matching for ranked lists in frame text:
   - Numbered: `1.`, `#1`, `No. 1`, `Number 1`, `1)`
   - Positioned: `TOP 1`, `BEST #1`
   - Handle multi-line: rank on one frame line, title on next
3. Each item: `rank`, `title`, `timestamp_secs`, `source_frame_index`
4. Deduplication across frames
5. Fallback to description then transcript if OCR yields fewer than 2 items
6. Pass enriched items to TMDb in pipeline

### 1.4 — Rewrite ShoppingExtractor

**File:** `backend/services/extraction/extractors.py`

**Required behavior:**
1. YOLO detections (object detection) remain primary for product category identification
2. OCR from frames is used to extract product NAMES and PRICES that appear as text overlays
3. Merge: for each YOLO-detected product category in a frame, check OCR text in that same frame for a matching name/price
4. Pattern: `Product Name - $XX.XX`, `Product Name | Price: $XX`, standalone product names
5. Each product: `name`, `brand`, `category`, `price`, `frame_index`, `timestamp_secs`, `yolo_confidence`, `search_url`

### 1.5 — Add RecipeExtractor

**File:** `backend/services/extraction/extractors.py`

**New extractor — does not exist yet:**

```python
class RecipeExtractor(BaseExtractor):
```

**Required behavior:**
1. OCR primary: extract ingredient lines from frames
2. Pattern matching:
   - `quantity unit ingredient` (e.g., "2 cups flour", "1 tsp salt")
   - Ingredient lists that appear as on-screen graphics
3. Step extraction from both OCR and transcript
4. Output: `{type: "recipe", title, ingredients: [{quantity, unit, name}], steps: [{index, text, timestamp_secs}], servings, prep_time, cook_time}`
5. Add `"recipe"` to `CATEGORIES` in `classifier.py` and training scripts (making N_CLASSES = 10)

### 1.6 — Update Factory Function

**File:** `backend/services/extraction/extractors.py`

```python
_EXTRACTOR_MAP = {
    "comedy": ComedyExtractor(),
    "listicle": ListicleExtractor(),
    "music": MusicExtractor(),
    "educational": EducationalExtractor(),
    "news": GenericExtractor(),
    "review": GenericExtractor(),
    "gaming": GenericExtractor(),
    "vlog": GenericExtractor(),
    "recipe": RecipeExtractor(),       # NEW
    "shopping": ShoppingExtractor(),
    "unknown": GenericExtractor(),
}
```

### 1.7 — Update Pipeline to Pass OCR Results to Extractors

**File:** `backend/services/pipeline.py`

After OCR runs (Step 3 in new order), pass `video_ocr_result` to every extractor call:

```python
extraction = extractor.extract(
    transcript_text=transcript_text,
    segments=transcript_segments,
    metadata=metadata,
    frame_paths=frame_result.frame_paths,
    frame_ocr_results=video_ocr_result.frames,  # NEW
)
```

### 1.8 — Store Per-Frame Data in MongoDB

**File:** `backend/services/pipeline.py` — `_persist_to_mongo()`

The MongoDB document must now include:

```json
{
  "analysis_id": "...",
  "video": {...},
  "classification": {...},
  "transcription": {...},
  "ocr_summary": {
    "total_frames": 60,
    "frames_with_text": 23,
    "aggregated_text": "..."
  },
  "frames": [
    {
      "frame_index": 0,
      "timestamp_secs": 1.2,
      "frame_path": "frame_00000.jpg",
      "ocr_text": "#20 Blinding Lights - The Weeknd",
      "has_content": true,
      "yolo_detections": []
    }
  ],
  "output": {...},
  "processing_time_secs": 87.3
}
```

### Completion Criteria for Phase 1:
- [x] MusicExtractor finds songs from frame OCR text
- [x] ListicleExtractor finds ranked items from frame OCR text
- [x] ShoppingExtractor merges YOLO + frame OCR
- [x] RecipeExtractor exists and works
- [x] Per-frame data stored in MongoDB
- [x] All extractors accept `frame_ocr_results` parameter
- [x] Test: submit a music countdown video with no speech → at least 5 songs found
- [x] Test: submit a movie list video → at least 3 movies found
- [x] All existing tests still pass

---

## PHASE 2 — TRANSLATION & MULTI-LANGUAGE SUPPORT
**Goal:** Full support for non-English videos. Whisper translation. Language-aware extraction.
**Duration:** 2 days
**Status:** [x] COMPLETE
**Prerequisite:** Phase 1 complete.

### 2.1 — Extend AudioTranscriber

**File:** `backend/services/audio_processor/transcriber.py`

**Add translation capability:**

```python
@dataclass
class TranscriptionResult:
    full_text: str
    full_text_english: Optional[str]   # NEW — translated if non-English
    language: str
    language_probability: float
    segments: List[Segment]
    segments_english: Optional[List[Segment]]  # NEW
    word_count: int
    was_translated: bool  # NEW
```

**New method:**
```python
async def transcribe_with_translation(
    self,
    audio_path: str,
    language: Optional[str] = None,
) -> TranscriptionResult:
```

**Logic:**
1. Run first transcription pass with `task="transcribe"` → gets native language text
2. Detect language from `info.language`
3. If `info.language != "en"` and `info.language_probability > 0.7`:
   - Run second pass with `task="translate"` → English translation
   - Store both in `TranscriptionResult`
4. If English already: skip second pass, `was_translated = False`

**Pipeline uses:** `transcript_english` for extraction. `transcript_original` stored for display. Frontend shows both with toggle.

### 2.2 — Language-Aware OCR

**File:** `backend/services/vision/ocr_service.py`

Tesseract supports multi-language OCR. When the detected audio language is non-Latin-script (Korean, Japanese, Chinese, Arabic, Hindi), OCR must use the appropriate Tesseract language pack:

```python
LANGUAGE_TO_TESSERACT = {
    "ko": "kor+eng",
    "ja": "jpn+eng",
    "zh": "chi_sim+chi_tra+eng",
    "ar": "ara+eng",
    "hi": "hin+eng",
    "th": "tha+eng",
    # default: "eng"
}
```

The detected audio language (from Whisper) should be passed to `OCRService` to select the correct Tesseract language pack.

### 2.3 — Update MongoDB Schema for Translation

The stored document must include:
```json
{
  "transcription": {
    "full_text": "원문 텍스트...",
    "full_text_english": "Original text translated to English...",
    "language": "ko",
    "was_translated": true
  }
}
```

### 2.4 — Update API Response Schema

**File:** `backend/models/schemas.py`

```python
class TranscriptResult(BaseModel):
    full_text: str
    full_text_english: Optional[str] = None
    language: str
    was_translated: bool = False
    segments: List[TranscriptSegment]
    word_count: int
```

### 2.5 — Frontend: Language Toggle

**File:** `frontend/src/App.jsx`

In `TranscriptPanel` component: if `transcription.was_translated == true`, show a toggle button "Show Original / Show English" that switches between `full_text` and `full_text_english`.

### Completion Criteria for Phase 2:
- [x] Korean/Japanese/Spanish/Hindi video produces English output
- [x] Both original and English transcript stored in MongoDB
- [x] Frontend toggle works for non-English videos
- [x] OCR uses correct Tesseract language pack based on detected language
- [x] Test: submit a Korean music countdown video → songs extracted in English

---

## PHASE 3 — FRAME SERVING & VISUAL TIMELINE UI
**Goal:** Serve extracted frames over HTTP. Build frame timeline component in frontend.
**Duration:** 3 days
**Status:** [x] COMPLETE
**Prerequisite:** Phase 2 complete.

### 3.1 — Static Frame File Server

**File:** `backend/main.py`

Add a static file serving route for extracted frames:

```python
from fastapi.staticfiles import StaticFiles

app.mount(
    "/frames",
    StaticFiles(directory=settings.FRAMES_DIR),
    name="frames"
)
```

Frame URL pattern: `http://localhost:8000/frames/{video_id}/frame_00012.jpg`

**Important:** Frames must NOT be deleted after analysis completes if the result is being served. Add a configurable `KEEP_FRAMES_AFTER_ANALYSIS: bool = True` setting. Only delete if false.

Frame files are stored in `{FRAMES_DIR}/{video_id}/frame_NNNNN.jpg` — this matches the existing `FrameExtractor` output structure.

### 3.2 — Frame URL in MongoDB Result

**File:** `backend/services/pipeline.py`

When storing the result, convert local frame paths to HTTP-accessible URLs:

```python
def _frame_path_to_url(frame_path: str, video_id: str) -> str:
    filename = Path(frame_path).name
    base = settings.PUBLIC_BASE_URL  # e.g., "http://localhost:8000"
    return f"{base}/frames/{video_id}/{filename}"
```

Add `PUBLIC_BASE_URL: str = "http://localhost:8000"` to `config.py`.

Each frame in the MongoDB document gets a `frame_url` field alongside `frame_path`.

### 3.3 — Frame Timeline API Endpoint

**File:** `backend/api/routes/analysis.py`

New endpoint:
```
GET /api/v1/analyses/{analysis_id}/frames
```

Returns:
```json
{
  "video_id": "...",
  "frames": [
    {
      "frame_index": 0,
      "timestamp_secs": 1.2,
      "frame_url": "http://localhost:8000/frames/abc123/frame_00000.jpg",
      "ocr_text": "#20 Blinding Lights - The Weeknd",
      "has_content": true,
      "yolo_detections": [
        {"label": "cell phone", "confidence": 0.87}
      ]
    }
  ]
}
```

Only returns frames where `has_content == true` OR `yolo_detections` is non-empty to avoid returning 60 empty frames.

### 3.4 — Frame Timeline Component (Frontend)

**File:** `frontend/src/App.jsx`

New tab in `ResultView`: **"Timeline"** tab (appears after "Output", "Classification", "Transcript").

Component: `FrameTimeline`

**Layout:**
- Horizontal scrollable timeline
- Each frame shown as a thumbnail (120px wide)
- Below each thumbnail: timestamp (e.g., "1:23")
- Below timestamp: OCR text found in that frame (truncated to 40 chars)
- Frames with no OCR text shown as greyed out (or hidden by default with "Show all frames" toggle)
- Clicking a frame opens a modal with full-size frame image + full OCR text + YOLO detections

**Only show this tab if `frames` array exists in the result and has at least one content frame.**

### 3.5 — Frame Modal Component

When a timeline frame is clicked:
```
┌─────────────────────────────────────────────┐
│  Frame 12 — 0:23                           ✕ │
├─────────────────────────────────────────────┤
│                                             │
│         [Full-size frame image]             │
│                                             │
├─────────────────────────────────────────────┤
│  OCR Text:                                  │
│  "#1 Blinding Lights - The Weeknd"         │
├─────────────────────────────────────────────┤
│  Detected Objects:                          │
│  📱 cell phone (87%)  💻 laptop (72%)       │
└─────────────────────────────────────────────┘
```

### Completion Criteria for Phase 3:
- [x] Frames served at `/frames/{video_id}/frame_NNNNN.jpg`
- [x] Frames not deleted if KEEP_FRAMES_AFTER_ANALYSIS is true
- [x] `/api/v1/analyses/{id}/frames` endpoint works
- [x] Frame timeline component renders in frontend
- [x] Frame modal opens on click with full image and OCR text
- [x] Test: submit a music countdown → timeline shows each song frame with text

---

## OUT OF BAND — USER MANAGEMENT (Bonus Phase)
**Goal:** Role-Based Access Control and Superuser CLI
**Status:** [x] COMPLETE

---

## PHASE 4 — SPOTIFY PLAYLIST QUALITY & AUTO-CREATION
**Goal:** Dramatically improve Spotify matching quality. Auto-create playlist on music analysis complete.
**Duration:** 2–3 days
**Status:** [x] COMPLETE
**Prerequisite:** Phase 3 complete.

### 4.1 — Improve Spotify Track Matching

**File:** `backend/services/integration/spotify_service.py`

**Current problem:** Searches `track:"Title" artist:"Artist"` — fails if OCR extracted typos, alternate formatting, or incomplete names.

**Required multi-strategy search:**

```python
async def search_track_robust(
    self,
    title: str,
    artist: str,
    ocr_raw: Optional[str] = None,  # original OCR text for fallback
) -> Optional[TrackInfo]:
```

**Strategy cascade (try in order, stop on first match):**
1. `track:"cleaned_title" artist:"cleaned_artist"` — exact match
2. `"cleaned_title" "cleaned_artist"` — quoted but not field-specific
3. `"cleaned_title" cleaned_artist` — only quote title
4. `cleaned_title cleaned_artist` — no quotes
5. `cleaned_title` — title only (no artist) + verify artist name similarity ≥ 70%
6. If `ocr_raw` provided: try searching the raw OCR line directly
7. Return `None` only if all 6 strategies fail

**Text cleaning before search:**
- Remove `#N`, `No. N`, `(N)` rank prefixes
- Remove year `(2020)`, `[2020]`
- Remove `(feat. X)`, `ft. X` — try with and without
- Remove special characters except letters, numbers, spaces, hyphens
- Normalize Unicode (smart quotes, em-dashes → standard)
- Strip `"`, `'` wrapping

### 4.2 — Auto-Create Playlist After Music Analysis

**File:** `backend/services/pipeline.py`

For music category: if Spotify is available AND user is authenticated AND at least 3 tracks found:
- Automatically create a Spotify playlist in the background
- Playlist name: `"{video_title} — YT Extracted"`
- Store playlist URL in the result under `output.spotify_playlist_url`
- This removes the manual "Create Playlist" step for the primary use case

For anonymous users (no Spotify token): store a `pending_playlist_tracks` list so the user can authenticate later and immediately get their playlist.

### 4.3 — Playlist Preview in Frontend

**File:** `frontend/src/App.jsx` — `MusicOutput` component

Show a prominent playlist card at the top of the music result:

```
┌────────────────────────────────────────────┐
│  🎵  Spotify Playlist Ready                │
│  20 tracks • Created from OCR extraction   │
│  [Open in Spotify ↗]  [Copy Link]          │
└────────────────────────────────────────────┘
```

If playlist was not auto-created (anonymous user):
```
┌────────────────────────────────────────────┐
│  🎵  Ready to Create Playlist              │
│  20 tracks identified                      │
│  [Connect Spotify & Create Playlist]       │
└────────────────────────────────────────────┘
```

### 4.4 — Track Confidence Indicators

Each track in the result should have a `match_confidence` field:
- `"exact"` — matched via strategy 1 or 2
- `"fuzzy"` — matched via strategy 3–5
- `"not_found"` — all strategies failed

In the UI, `not_found` tracks are shown with a grey "Search manually →" link instead of a Spotify link.

### Completion Criteria for Phase 4:
- [x] Multi-strategy Spotify search implemented and tested
- [x] Playlist auto-created for authenticated music analysis
- [x] `spotify_playlist_url` in result for authenticated users
- [x] Playlist preview card in frontend (`MusicOutput` component)
- [x] Track confidence indicators (`exact`, `fuzzy`, `not_found`) in result and UI
- [x] Test: cascading multi-strategy search raises match rate significantly

---

## PHASE 5 — TMDB ENRICHMENT & LISTICLE QUALITY
**Goal:** Full TMDb enrichment for movie/TV lists. Streaming availability. Accurate listicle extraction.
**Duration:** 2 days
**Status:** [x] COMPLETE
**Prerequisite:** Phase 4 complete.

### 5.1 — Improve Listicle Text Cleaning

**File:** `backend/services/extraction/extractors.py` — `ListicleExtractor`

OCR text from frames often contains noise around the actual title. Common patterns from real YouTube videos:
- `#1 THE SHAWSHANK REDEMPTION (1994)` → `The Shawshank Redemption` + year `1994`
- `No. 5\nPulp Fiction` (rank and title on separate lines in same frame)
- `BEST MOVIE EVER\nThe Godfather` (label text before title)
- `The Dark Knight [2008]` → title + year in brackets

Add a `_clean_listicle_title()` function that:
1. Strips rank prefix (`#N`, `No. N`, `N.`, `N)`)
2. Extracts year if present (4-digit number in parens/brackets) → store as `year` field
3. Title-cases the result
4. Strips noise words: "THE BEST", "GREATEST EVER", "MUST WATCH"
5. Returns `(cleaned_title, extracted_year)`

### 5.2 — TMDb Search with Year Hint

**File:** `backend/services/integration/tmdb_service.py`

When year is extracted from OCR text alongside the title, pass it to TMDb search as a disambiguation hint. This significantly improves accuracy for common movie titles.

Add TV show fallback: if movie search returns no results, try TV show search with same title.

### 5.3 — Streaming Availability Display

**File:** `frontend/src/App.jsx` — `ListicleOutput` component

For each movie item with streaming data:
- Show platform badges: Netflix (red), Prime (blue), Disney+ (dark blue), Hulu (green)
- Show "Available on N platforms" summary
- "Rent" and "Buy" options as secondary indicators

### Completion Criteria for Phase 5:
- [x] Listicle title cleaning extracts years and strips rank prefixes from OCR text
- [x] TMDb search uses year hint when available
- [x] TV show fallback in TMDb search (if movie search fails)
- [x] Streaming availability badges (Netflix, Prime, Disney+, Hulu) in frontend
- [x] Test: `_clean_listicle_title` unit tests pass

---

## PHASE 6 — RECIPE EXTRACTION POLISH
**Goal:** Recipe extraction works reliably. Structured output. Scaling.
**Duration:** 2 days
**Status:** [x] COMPLETE
**Prerequisite:** Phase 5 complete.

### 6.1 — Ingredient Parser

**File:** `backend/services/extraction/extractors.py` — `RecipeExtractor`

Ingredient pattern recognition:
- `2 cups flour` → `{quantity: 2, unit: "cups", name: "flour"}`
- `1/2 tsp salt` → `{quantity: 0.5, unit: "tsp", name: "salt"}`
- `3 large eggs` → `{quantity: 3, unit: null, name: "large eggs"}`
- `A pinch of pepper` → `{quantity: null, unit: "pinch", name: "pepper"}`

Unit normalization dictionary: `tsp`/`teaspoon` → `tsp`, `tbsp`/`tablespoon` → `tbsp`, `cup`/`cups` → `cup`, etc.

### 6.2 — Recipe Frontend Component

**File:** `frontend/src/App.jsx`

New `RecipeOutput` component:
- Ingredient list with checkboxes (user can tick off as they cook)
- Serving size scaler: adjust servings → all quantities scale proportionally
- Step-by-step accordion
- "Copy ingredient list" button (clipboard)
- Each ingredient links to a Google Shopping search

### Completion Criteria for Phase 6:
- [x] Ingredient parser handles fractions, decimals, lists, and pinch/a amounts
- [x] Unit normalization dict maps verbose unit names to standard short forms
- [x] Serving scaler works in UI (ratio-based quantity scaling)
- [x] Recipe component renders with checkboxes, accordion steps, copy button, and shopping links
- [x] Unit tests for ingredient parser pass (`tests/unit/test_recipe_parsing.py`)

---

## PHASE 7 — INFRASTRUCTURE & PRODUCTION HARDENING
**Goal:** Docker Compose, CI/CD, monitoring, security hardening. Production ready.
**Duration:** 3–4 days
**Status:** [x] COMPLETE
**Prerequisite:** Phase 6 complete.

### 7.1 — Docker Compose

**File:** `docker-compose.yml` (create at project root)

Services:
```yaml
services:
  postgres:     postgres:16-alpine, port 5432, volume postgres_data
  mongodb:      mongo:7, port 27017, volume mongo_data
  redis:        redis:7-alpine, port 6379, volume redis_data
  backend:      Dockerfile at backend/, port 8000
  worker:       Same Dockerfile, command: celery worker
  flower:       mher/flower, port 5555, monitors Celery
  frontend:     Dockerfile at frontend/, port 3000
```

Health checks on all services. `depends_on` with health conditions.

### 7.2 — Dockerfile (Backend)

**File:** `backend/Dockerfile`

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y ffmpeg tesseract-ocr tesseract-ocr-eng libgl1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm
COPY . /app
WORKDIR /app
```

### 7.3 — Dockerfile (Frontend)

**File:** `frontend/Dockerfile`

Multi-stage build: build with Node → serve with nginx.

### 7.4 — Environment Template

**File:** `.env.example` (create at project root)

Document every variable from `config.py` with examples and comments. Include `openssl rand -hex 32` instruction for SECRET_KEY.

### 7.5 — GitHub Actions CI

**File:** `.github/workflows/ci.yml`

- Install Python deps
- Run `pytest tests/ -v --asyncio-mode=auto`
- Install Node deps
- Run `npm run build` (frontend)
- Report pass/fail

### 7.6 — Sentry Integration

Already partially in `main.py`. Verify `SENTRY_DSN` env var is documented. Ensure Celery task errors are also captured by Sentry.

### 7.7 — Request Correlation IDs

Already in `main.py` as `CorrelationIDMiddleware`. Verify the `X-Request-ID` header appears in all responses and is logged in every log line.

### 7.8 — Refresh Token Blacklist on Logout

Already in `auth.py` — `logout` endpoint adds token to Redis blacklist. Verify `get_current_user` checks the blacklist on every request.

### Completion Criteria for Phase 7:
- [x] `docker-compose.yml` defines all 7 services with health checks and volumes
- [x] `docker-compose up` → frontend accessible at localhost:3000 (Nginx)
- [x] `docker-compose up` → API accessible at localhost:8000
- [x] Celery workers visible in Flower at localhost:5555 (mher/flower)
- [x] CI pipeline (`.github/workflows/ci.yml`) passes on push
- [x] `.env.example` documents all variables (with SECRET_KEY generation instruction)
- [x] Sentry integrated for both FastAPI (production lifespan) and Celery (`@worker_process_init`)
- [x] `X-Request-ID` added to all responses; `logger.contextualize` threads it through every log line

---

## PHASE 8 — TRAINING DATA & MODEL IMPROVEMENT
**Goal:** Build labeling tool. Collect 500 samples. Train text classifier. Improve accuracy.
**Duration:** 1 week (includes data collection time)
**Status:** [x] COMPLETE
**Prerequisite:** Phase 7 complete. System running in Docker.

### 8.1 — Labeling Tool (Frontend)

**File:** `frontend/src/App.jsx`

New page: `/label` (admin only)

Shows samples from `training_data/` directory where `human_label` is null:
- Video title
- Tags
- Pipeline prediction + confidence
- Transcript preview
- OCR text preview
- 10 category buttons (click to label)
- "Skip" button
- Progress indicator: "X / Y labeled"

On label click: POST to `/api/v1/admin/label` which writes `human_label` to the JSON file.

### 8.2 — Admin Labeling Endpoint

**File:** `backend/api/routes/analysis.py` or new `admin.py`

```
POST /api/v1/admin/label
Body: {sample_id: str, human_label: str}
```

Requires admin auth. Writes to `training_data/{sample_id}.json`.

### 8.3 — Training Data Export

```
GET /api/v1/admin/training-data/export
```

Returns CSV with all labeled samples. Used to feed `train_text_classifier.py`.

### 8.4 — Model Training Instructions

Document in `scripts/README.md`:
1. How to export labeled data
2. How to run text classifier training (CPU path: 5 epochs, ~4 hours)
3. How to run frame classifier training (requires GPU: recommend Colab A100)
4. How to place checkpoints in `backend/checkpoints/`
5. How to restart workers to load new weights

### Completion Criteria for Phase 8:
- [x] Labeling tool accessible at `/label` for admin users (phase-gated in App.jsx)
- [x] Admin nav item (`🏷️ Label`) visible only to admin users
- [x] `POST /api/v1/admin/label` writes `human_label` to training sample JSON
- [x] `GET /api/v1/admin/training-data` lists all samples with labeling status
- [x] `GET /api/v1/admin/training-data/export` streams a labeled-only CSV
- [x] Schema validation: `LabelRequest` rejects unknown categories, normalizes case
- [x] Unit tests added in `tests/unit/test_admin_routes.py`
- [x] `scripts/README.md` documents full training workflow (export → train → deploy)

---

## CROSS-CUTTING CONCERNS

These apply to every phase:

### Error Handling
Every async function must have try/except. Errors must be logged with `logger.error()` and include the `analysis_id` for tracing. No bare `except:` — always catch specific exceptions or at minimum `Exception as exc`.

### Logging
Every significant step in the pipeline must log at INFO level with `[{analysis_id}]` prefix. Use `logger.info(f"[{analysis_id}] Step X complete — found N items")`. Errors at ERROR level with full stack trace.

### Testing
Every new feature needs a test. Unit tests in `tests/unit/`. Integration tests in `tests/integration/`. Mock all external services (Spotify, TMDb). Tests must not require real API keys to run.

### Database Migrations
Any change to the PostgreSQL schema requires a new Alembic migration. Never modify existing migrations. Run `alembic revision --autogenerate -m "description"` to generate.

### Configuration
Every new configuration value goes in `core/config.py` as a typed field with a sensible default. Never hardcode URLs, paths, thresholds, or credentials anywhere else.

---

## BUILD STATUS TRACKER

Update this table after completing each phase:

| Phase | Name | Status | Completed Date | Notes |
|---|---|---|---|---|
| 0 | Foundation Hardening | ✅ Complete | (prior run) | Bugs fixed, prepped for frame-first. |
| 1 | Frame-First Extraction | ✅ Complete | (prior run) | OCR used as primary source. |
| 2 | Translation & Multi-Language | ✅ Complete | (prior run) | Whisper and pytesseract language integration. |
| 3 | Frame Serving & Timeline UI | ✅ Complete | Today | FastAPI static serve, React Timeline. |
| _ | Admin User Management | ✅ Complete | Today | Out of band phase (role, CLI). |
| 4 | Spotify Quality & Auto-Create | ✅ Complete | Today | Cascading search, Auto-playlist generation |
| 5 | TMDb Enrichment & Listicle | ✅ Complete | Today | Year hint extraction, TV show fallback, streaming badges |
| 6 | Recipe Extraction | ✅ Complete | Today | — |
| 7 | Infrastructure & Production | ✅ Complete | Today | Docker, CI/CD, Sentry, Security |
| 8 | Training Data & Model | ✅ Complete | Today | Admin labeling tool, export API, scripts/README.md |
| 9 | Gap Closing (context.md) | ✅ Complete | Today | Google Books, book-list UI, per-item provenance timestamps |

---

## CHANGE LOG

| Date | Phase | Change | Reason |
|---|---|---|---|
| — | — | Initial document created | — |
| Today | 0-3 | Completed mapping of core visual intelligence | Feature Implementation |
| Today | 4 | Add Admin Role + Superuser script | Out-of-band security fix |
| Today | 4 | Improved Spotify Track Matching & Auto-Playlists | Improved search strategy, UI integration, and token usage |
| Today | 5 | TMDb Enrichment & Listicle Quality | Extracted OCR year hints, TV fallback, UI streaming badges |
| Today | 6 | Recipe Extraction Polish | Scalable ratio quantities, float ingredients, standard unit mapping, Interactive Recipe Component |
| Today | 7 | Infrastructure & Production Hardening | Dockerized full stack, configured GitHub Actions CI, setup Sentry & Request Correlations |
| Today | 9 | Google Books Integration | Book-list detection, Books API enrichment, ISBN + author + Goodreads/Amazon links |
| Today | 9 | Per-item OCR Provenance | "found at 1:23 via OCR" timestamp badge on every Music and Listicle item in UI |

---

## PHASE 9 — GAP CLOSING (context.md Compliance)

**Goal:** Implement all features explicitly required by `context.md` that were not covered by Phases 0–8.
**Status:** [x] COMPLETE

### 9.1 — Google Books API Integration

**File:** `backend/services/integration/google_books_service.py`

- `GoogleBooksService.search_book(title, author?)` — async, Redis-cached (24hr TTL)
- Returns `BookInfo`: title, authors, description, ISBN, thumbnail, google_books_url, goodreads_url, amazon_url, published_date, page_count
- Gracefully returns `None` when `GOOGLE_BOOKS_API_KEY` is not set (degraded mode)

**Configuration:**
- `GOOGLE_BOOKS_API_KEY: Optional[str] = None` added to `core/config.py`
- Documented in `.env.example`

### 9.2 — Book-List Detection in ListicleExtractor

**File:** `backend/services/extraction/extractors.py`

- `ListicleExtractor._is_book_list(metadata)` — detects book-list videos via title/tag/description regex
- `ListicleExtractor.extract()` — branches based on `is_book_list`:
  - Book items: `item_type: "book"` with `authors`, `isbn`, `thumbnail`, `goodreads_url`, `amazon_url`, `google_books_url`, `published_date`, `page_count`
  - Media items: `item_type: "media"` with TMDb fields (unchanged)
- `extraction["is_book_list"]` field in pipeline result

**Pipeline:** `backend/services/pipeline.py`
- Enrichment step branches: if `is_book_list` → `google_books_service`, else → `TMDbService`

### 9.3 — Per-Item OCR Provenance in UI

**File:** `frontend/src/App.jsx`

- `ListicleOutput`: shows `🎬 found at 0:45 via OCR` badge per item when `timestamp_secs` is present
- `MusicOutput`: shows `🎬 1:23` timestamp per track when `timestamp_secs` is present
- Satisfies context.md §6 Innovation 5: "every extracted item is linked to the exact frame (and timestamp) where it was found"

### 9.4 — Book Card UI in Frontend

**File:** `frontend/src/App.jsx`

- `ListicleOutput` detects `is_book_list` and renders book-specific fields:
  - Amber rank badge (vs. blue for movies)
  - Book cover thumbnail from Google Books
  - Author line in amber (`by James Clear`)
  - Published date + page count
  - Goodreads ↗, Amazon ↗, Google Books ↗ links
  - ISBN tag

### Completion Criteria for Phase 9:
- [x] `google_books_service.py` created with Redis cache and graceful degradation
- [x] `GOOGLE_BOOKS_API_KEY` in config and `.env.example`
- [x] `_is_book_list()` heuristic detects book-list videos
- [x] Book items carry `item_type: "book"` with full enrichment fields
- [x] Pipeline correctly branches to Google Books vs TMDb based on `is_book_list`
- [x] Every Music track shows "🎬 1:23" timestamp provenance badge
- [x] Every Listicle item shows "🎬 found at 0:45 via OCR" provenance badge
- [x] Book card UI: cover, author, ISBN, Goodreads/Amazon/Google Books links
- [x] Frontend build passes clean (`vite build` ✓ 30 modules)
