# YTCLFR — THE COMPLETE ROADMAP
## From Broken Pipeline to MNC-Level Tier 3 Product

> **The Honest Truth:** The infrastructure is built. The database is running. The API exists. The frontend renders. What is missing is the brain and the correct data flow. This roadmap gives you the exact sequence of work — month by month — from today until you have a genuinely production-grade Tier 3 system.
>
> **Do not skip months. Do not reorder phases. Every month builds on the previous one.**

---

## The Full Picture (12 Months at a Glance)

| Phase | Month | Goal |
|-------|-------|------|
| **TIER 1** | Month 1 | **MAKE IT WORK** — Brain transplant + OCR fix |
| **TIER 1** | Month 2 | **MAKE IT RIGHT** — Full pipeline rewrite |
| **TIER 1** | Month 3 | **MAKE IT COMPLETE** — All features working |
| **TIER 1** | Month 4 | **MAKE IT RELIABLE** — Production hardening |
| **TIER 2** | Month 5–6 | **MAKE IT SMART** — First trained model |
| **TIER 3** | Month 7–12 | **MAKE IT YOURS** — Full custom ML system |

---

## The Three Metrics That Determine Success

At every milestone, measure these three numbers. All three green = MNC-level Tier 3.

| Metric | Current | Month 1 Target | Month 6 Target | Month 12 Target |
|--------|---------|----------------|----------------|-----------------|
| 🎵 Music Extraction Accuracy | ~40% | 75% | 83% | **≥ 88%** |
| ✅ End-to-End Success Rate | ~30% | 80% | 92% | **≥ 96%** |
| 🧠 LLM Dependency Rate | 100% | 100% | 25% | **≤ 10%** |

> **The North Star Test (run every week):**
> *"If I submit 'Top 20 Songs of 2020' with no narration and songs shown on screen — does the system return a Spotify playlist with all 20 songs?"*
> - Week 1: Answer becomes **YES** for the first time
> - Week 4: Answer is **YES** with 18+ songs on Spotify
> - Month 6: Answer is **YES** in under 90 seconds with a custom model
> - Month 12: Answer is **YES** in under 60 seconds, any language, any format

---

---

# MONTH 1 — MAKE IT WORK
## Goal: One URL in. Real result out. Every time.

---

## Week 1 — The Brain Transplant
> **Giving the system intelligence from day one**

Everything else in the pipeline is meaningless if the brain is broken. Classification is wrong. Extraction returns nothing useful. Fix the brain first.

### Tasks

**Day 1–2: Create the intelligence layer**

New directory: `backend/services/intelligence/`

| File | Purpose |
|------|---------|
| `__init__.py` | Package init |
| `llm_brain.py` | Claude API brain — takes OCR + title + tags + transcript, returns structured JSON with category AND all extracted items |
| `router.py` | Tier routing logic |
| `training_collector.py` | Auto-saves every result as training data |

**Day 2–3: Wire into `pipeline.py`**

Remove the two broken calls:
```python
# OLD (broken — remove both)
classification = await self._classify(...)
extraction = await self._extract(...)
```

Replace with one unified call:
```python
# NEW — one call that does both correctly
brain_result = await self._run_intelligence_layer(
    metadata=metadata,
    frame_ocr_results=frame_ocr_results,
    transcript_english=transcript_english,
    analysis_id=analysis_id,
)
```

**Day 3: Add config variables**

In `backend/core/config.py`:
```python
ANTHROPIC_API_KEY: str = ""
BRAIN_MODEL: str = "claude-haiku-4-5"
BRAIN_CONFIDENCE_THRESHOLD: float = 0.85
LLM_FALLBACK_ENABLED: bool = True
```

In `requirements.txt`:
```
anthropic>=0.40.0
```

**Day 4–5: Test the brain**

Submit these 5 test videos (real YouTube URLs):
1. "Top 20 Songs of 2020" — music, silent, text on screen
2. "Top 10 Movies of All Time" — listicle, text on screen
3. Any recipe video — ingredients shown on screen
4. A shopping haul video — product names shown
5. Korean or Spanish music countdown — tests non-English

For each, verify:
- [ ] Category is correct
- [ ] Items are extracted (not empty)
- [ ] Items came from OCR text, not random guesses
- [ ] Training data JSON saved to `training_data/`

### Done When
Submit "Top 20 Songs of 2020 (No Narration)" with songs on screen → system returns 20 song objects with title and artist. No empty results.

---

## Week 2 — Fix the OCR Foundation
> **Making sure the brain gets complete input**

OCR currently runs on 8–20 frames max. A 10-minute music countdown has 60+ frames. You are feeding the brain partial data.

### Tasks

**Day 1–2: Rewrite `ocr_service.py`**

- Remove `max_frames` limit from extraction path
- OCR must run on **every** frame
- Store as `List[FrameOCRData]`:

```python
@dataclass
class FrameOCRData:
    frame_index: int
    frame_path: str
    timestamp_secs: float
    raw_text: str
    cleaned_text: str
    confidence: float       # Tesseract word confidence 0–100
    has_content: bool       # True if 3+ meaningful words found
```

- Add semaphore to prevent memory crash: `asyncio.Semaphore(8)` — max 8 frames OCR at once

**Day 2–3: Fix the pipeline order**

| | Current (Wrong) | Required (Correct) |
|--|--|--|
| Order | download → frames → transcribe → classify → OCR → extract | download → frames → **OCR ALL FRAMES** → transcribe → **BRAIN** → enrich |

**Day 3–4: Store per-frame data in MongoDB**

```json
{
  "frames": [
    {
      "frame_index": 12,
      "timestamp_secs": 23.4,
      "frame_url": "http://localhost:8000/frames/abc123/frame_00012.jpg",
      "ocr_text": "#1 Blinding Lights - The Weeknd",
      "has_content": true,
      "yolo_detections": []
    }
  ]
}
```

**Day 5: Verify improvement**

Same "Top 20 Songs" test video from Week 1.
- Before: 14–16 songs (missing frames)
- After: **all 20 songs, every time**

---

## Week 3 — Translation & Silent Video Handling
> **Making the system work for the whole world**

Millions of the best music countdown videos are in Korean, Spanish, Hindi, Japanese. Whisper already supports translation — it is just not being used.

### Tasks

**Day 1–2: Extend `transcriber.py`**

```python
@dataclass
class TranscriptionResult:
    full_text: str                        # original language
    full_text_english: Optional[str]      # translated to English
    language: str                         # detected language code
    language_probability: float
    was_translated: bool
    segments: List[Segment]
```

Logic:
1. First pass: `task="transcribe"` → detect language, get original text
2. If `language != "en"` and `confidence > 0.7`: second pass with `task="translate"`
3. Store both. Extraction always uses `full_text_english`.

**Day 2–3: Language-aware OCR**

```python
LANG_MAP = {
    "ko": "kor+eng",
    "ja": "jpn+eng",
    "zh": "chi_sim+chi_tra+eng",
    "ar": "ara+eng",
    "hi": "hin+eng",
}
```

**Day 3–4: Handle silent video case explicitly**

When `transcript_text` has fewer than 20 words:
```python
logger.info(f"[{analysis_id}] Silent/no-narration video detected. OCR is the only data source.")
```

Add a test: completely silent video still returns results.

**Day 5: Multi-language test**

Submit a Korean music countdown, a Spanish movie list, a Japanese recipe. Verify English output with correct items.

---

## Week 4 — Enrichment Layer Verification
> **Making sure Spotify and TMDb actually work end to end**

The brain now extracts clean song names and movie titles. But the enrichment layer has bugs that were invisible before extraction was fixed.

### Tasks

**Day 1–2: Fix Spotify track matching**

Implement a 6-strategy cascade (stop at first match):

| Strategy | Query |
|----------|-------|
| 1 | `track:"cleaned_title" artist:"cleaned_artist"` (exact) |
| 2 | `"cleaned_title" "cleaned_artist"` (quoted) |
| 3 | `"cleaned_title" cleaned_artist` (title only quoted) |
| 4 | `cleaned_title cleaned_artist` (no quotes) |
| 5 | `cleaned_title` alone + verify artist similarity |
| 6 | Raw OCR line directly |

Text cleaning before search:
- Remove rank prefix: `#1`, `No. 1`, `1.`
- Remove year: `(2020)`, `[2020]`
- Remove feat: `(feat. X)`, `ft. X` → try with and without

**Day 2–3: Auto-create Spotify playlist**

- After music analysis completes, auto-create playlist named `"{video_title} — YTCLFR"`
- Store `spotify_playlist_url` in result
- For unauthenticated users: store `pending_tracks` for when they connect later

**Day 3–4: Fix TMDb search**

- Pass extracted year as a hint: `"The Dark Knight (2008)"` → use year in TMDb query
- Add TV show fallback: if movie search returns nothing, try TV search

**Day 5: End-to-end test**

Submit "Top 20 Songs of 2020". Verify:
- [ ] 20 songs extracted from OCR
- [ ] 18+ found on Spotify
- [ ] Playlist created automatically for authenticated user
- [ ] Playlist URL returned in result
- [ ] Playlist opens in Spotify and plays correctly

**This is the north star moment. If this works, Month 1 is complete.**

---

---

# MONTH 2 — MAKE IT RIGHT
## Goal: Every feature works correctly, not just the happy path

---

## Week 5 — Frame Timeline & Visual Proof
> **Showing users exactly what the system found and where**

Users do not trust what they cannot see. Visual provenance is the feature that separates YTCLFR from every other tool.

### Tasks

**Day 1: Static frame file server**

Add to `backend/main.py`:
```python
app.mount("/frames", StaticFiles(directory=settings.FRAMES_DIR), name="frames")
```

New config:
```python
FRAMES_DIR: str = "/tmp/ytclassifier/frames"
KEEP_FRAMES_AFTER_ANALYSIS: bool = True
PUBLIC_BASE_URL: str = "http://localhost:8000"
```

**Day 2: Frames API endpoint**

```
GET /api/v1/analyses/{analysis_id}/frames
```

Returns only frames where `has_content == true` or YOLO detected something.

**Day 3–5: Frame timeline component in React**

New "Timeline" tab in ResultView:
- Horizontally scrollable strip of frame thumbnails
- Below each: timestamp formatted as `"0:23"`
- Below timestamp: first 35 characters of OCR text found
- Greyed out: frames with no content detected
- Click any frame: full-size modal with image + OCR text + YOLO boxes

---

## Week 6 — Category-Specific Output Components
> **Making each category's output actually useful**

Right now all results show the same generic JSON dump. Each category needs its own purpose-built UI component.

### Output Components

**🎵 Music**
- Spotify playlist banner with "Open in Spotify" + "Copy Link" buttons
- Numbered track list with artist, confirmation status (✓ / ?)

**🎬 Listicle**
- Movie/book/game cards with poster image, TMDb rating
- Streaming badges (Netflix, Prime, Disney+, Hulu) as colored pills

**🍳 Recipe**
- Ingredient list with checkboxes
- Serving size scaler slider (2–8 servings), quantities scale proportionally in real time
- "Copy ingredient list" button

**🛍 Shopping**
- Product name, detected brand, price if found
- "Search on Google Shopping" link per item
- YOLO-detected product category badge

---

## Week 7 — Error Handling & Reliability
> **Making sure the system never silently fails**

This week separates a demo project from a production system.

### Failure Modes & Handlers

| Failure | Response |
|---------|----------|
| OCR fails on a frame | Log error + frame path, continue with remaining frames, never crash pipeline |
| Claude API down / rate limited | Retry 3× with exponential backoff (1s, 4s, 16s) → fall back to regex heuristics → set status to `"partial"` |
| Spotify API fails | Return items with `spotify_status: "search_failed"`, allow retry separately |
| Video download fails | Return specific error code: `VIDEO_PRIVATE`, `VIDEO_DELETED`, `VIDEO_AGE_RESTRICTED` |
| Celery task crashes midway | Retry with `max_retries=3`, save checkpoint to Redis after each major step, resume from checkpoint |

---

## Week 8 — Docker & CI/CD
> **Making the system deployable anywhere**

**Day 1–2: `docker-compose.yml` at project root**

Services: `postgres`, `mongodb`, `redis`, `backend` (FastAPI), `worker` (Celery), `flower` (Celery monitor), `frontend` (React/nginx).
- Health checks on every service
- `depends_on` with health conditions
- Volumes for data persistence

**Day 3: Verify full stack in Docker**

`docker-compose up` → submit a URL → get a result. No manual steps. One command.

**Day 4–5: GitHub Actions CI**

On every push:
1. `pip install -r requirements.txt`
2. `pytest tests/ -v --asyncio-mode=auto`
3. `npm install && npm run build`
4. Report pass/fail

Every PR must pass CI before merge.

---

---

# MONTH 3 — MAKE IT COMPLETE
## Goal: Every advertised feature works. No known gaps.

---

## Week 9–10 — Recipe Extraction Polish

Recipe videos present data in many formats: ingredients one-by-one as graphic cards, block lists at start, narrated steps with no text overlay, numbered text on screen.

- **Week 9:** Build ingredient parser with unit normalization. Handles: `"2 cups"`, `"1/2 tsp"`, `"3 large eggs"`, `"a pinch of"`
- **Week 10:** Build serving scaler in frontend. All ingredient amounts stored as numbers, multiplied by `currentServings / baseServings` on render.

---

## Week 11 — Admin Labeling Tool
> **Building the data engine for Tier 2**

Admin page at `/label` (requires admin login):

```
Progress: 47 / 500 labeled

┌─────────────────────────────────────────────────┐
│  Video: "Best Movies of the Decade"             │
│  LLM said: listicle (confidence: 0.94)         │
│                                                 │
│  OCR preview:                                   │
│  "#10 Parasite (2019)"                         │
│  "#9 Marriage Story (2019)"                    │
│                                                 │
│  What is this video?                            │
│  [music] [listicle] [recipe] [shopping]        │
│  [educational] [gaming] [vlog] [news] [review] │
│                                                 │
│              [Skip]    [Confirm LLM Label ✓]   │
└─────────────────────────────────────────────────┘
```

At 10 seconds per sample: 500 labels ≈ 83 minutes. One afternoon.

---

## Week 12 — Analytics & Monitoring Dashboard

**System metrics:**
- Videos analyzed today / this week / total
- Average processing time per video
- Success rate (completed / attempted)
- Category distribution
- LLM call count + cost estimate

**Quality metrics:**
- Spotify match rate (found tracks / extracted tracks)
- TMDb match rate (found movies / extracted movies)
- Average items per video per category
- Videos where extraction returned 0 items (failure cases)

> You cannot improve what you do not measure. If Spotify match rate is 70%, matching logic needs work. If recipe extraction returns 0 items 40% of the time, OCR pattern matching for recipes is broken.

---

---

# MONTH 4 — MAKE IT RELIABLE
## Goal: The system handles load, failures, and edge cases without human intervention

---

## Week 13–14 — Production Hardening

**Structured logging with correlation IDs:**
```python
logger.info(
    "OCR complete",
    extra={
        "analysis_id": analysis_id,
        "frames_processed": 60,
        "frames_with_content": 23,
        "duration_ms": 4200,
    }
)
```

**Dead letter queue:** Celery tasks that fail after all retries go to DLQ. Admin can inspect and replay. Nothing is silently lost.

**Database connection pooling:**
- PostgreSQL: max 20 connections
- MongoDB: max 10 connections

**Rate limiting:** Verify existing Redis sliding window rate limiter works under load.
- Free tier: 5 analyses/hour
- Authenticated: 50 analyses/hour

---

## Week 15–16 — Performance Optimization

**Parallel OCR + transcription:**
```python
ocr_task = asyncio.create_task(self._run_ocr(frame_paths))
transcription_task = asyncio.create_task(self._run_transcription(audio_path))
ocr_results, transcription = await asyncio.gather(ocr_task, transcription_task)
```
Expected improvement: **30–40% reduction in total pipeline time.**

**Frame extraction optimization:** Scene detection to avoid extracting 5 nearly identical frames within 2 seconds. Adaptive sampling: fast for short videos, slower for long ones.

**Result caching:** Same YouTube URL submitted within 24 hours → return cached result from MongoDB. Do not re-download. Hash the URL as cache key.

---

---

# MONTH 5–6 — MAKE IT SMART (TIER 2)
## Goal: Your first custom trained model. 70–80% cost reduction.

---

## Month 5 — Collect and Prepare Training Data

By now the system has processed hundreds of videos. `training_data/` has hundreds of auto-labeled JSON files.

**Week 1–2: Export and clean**
```
GET /api/v1/admin/training-data/export
```
Expected: 500+ samples across 9 categories. Check balance: at least 40 samples per category. Manually submit more if any category is underrepresented.

**Week 3–4: Prepare datasets**

Run `scripts/prepare_dataset.py`:
- Input: all `training_data/*.json` files where `llm_label` is set
- Output: `data/text_classifier_dataset.csv` (title, tags, ocr_aggregated_text, transcript_preview, label)
- Output: `data/frame_classifier_dataset.csv` (frame_path, label)
- Split: **80% train / 10% validation / 10% test** — fixed random seed

---

## Month 6 — Train, Evaluate, Deploy

**Train text classifier (CPU — your machine, ~2–4 hours):**
```bash
python scripts/train_text_classifier.py \
  --data data/text_classifier_dataset.csv \
  --model distilbert-base-uncased \
  --epochs 5 \
  --batch-size 16 \
  --lr 2e-5 \
  --output backend/checkpoints/best_text_model.pth
```
Expected accuracy: **78–85%**

**Train frame classifier (GPU — Colab A100, ~$3–5, 20–30 min):**
```bash
python scripts/train_frame_classifier.py \
  --data data/frame_classifier_dataset.csv \
  --model efficientnet_b0 \
  --epochs 20 \
  --batch-size 32 \
  --lr 1e-4 \
  --device cuda \
  --output backend/checkpoints/best_frame_model.pth
```
Expected accuracy: **65–72%**

**Deploy Tier 2 routing in `router.py`:**
- DistilBERT confidence ≥ 0.85 → use local result, skip LLM classify call
- Confidence < 0.85 → fall back to LLM

**Cost impact:**
| Before Tier 2 | After Tier 2 |
|---|---|
| Every video = 1 full LLM call | 80% of videos = 0 LLM classify calls |
| ~$0.002 per video | ~$0.0006 per video |
| **Reduction: ~70%** | |

---

---

# MONTH 7–12 — MAKE IT YOURS (TIER 3)
## Goal: Your own brain. LLM is the fallback, not the primary.

---

## Month 7–8 — Custom Extraction Pattern Library

By Month 7 you have processed 2000+ videos. You know the exact OCR text patterns from real data.

Build a comprehensive pattern matcher trained on real data:
- **Music:** `"#N Title - Artist"`, `"N. Title by Artist"`, `"Title | Artist"`, `"Artist - Title (Year)"` — every variation seen in real data
- **Listicle:** Same approach — real data tells you every format

These patterns replace LLM extraction for confident cases. LLM extraction runs only for genuinely novel formats.

---

## Month 9–10 — MLOps Layer

**Model registry:** Every trained model version stored in a registry (S3 + `versions.json` manifest). Rollback = change version reference.

**A/B testing:** New model → 10% of traffic first. Compare accuracy against production. Promote to 100% only if it improves accuracy.

**Drift monitoring:** Track category distribution over time. Alert if any category shifts more than 15% in a week.

**Automated retraining pipeline:**
1. Export new training samples
2. Merge with existing data
3. Fine-tune from current model weights (not from scratch)
4. Evaluate against held-out test set
5. Promote if improved, discard if not

Run monthly. Model continuously improves.

---

## Month 11–12 — Enterprise Features

**Multi-store shopping:**
```python
class RetailerAdapter(ABC):
    @abstractmethod
    async def search(self, product_name: str) -> List[ProductResult]:
        pass
```
Implement: `GoogleShoppingAdapter`, `AmazonAdapter` (PAAPI), `BestBuyAdapter`. Price comparison: same product across stores side by side.

**User history & collections:**
- Save analyses to named collections: "My Movie Watchlist", "Recipes I Saved", "Music Playlists"

**Sharing:**
- Public shareable URL: `ytclfr.com/share/abc123`
- SEO-friendly: title, category, item count in meta tags

**Developer API:**
- Public API with API key auth
- Rate limited + documented
- Developers submit URLs, get structured JSON back
- Separate revenue stream from the consumer product

---

---

# Week-by-Week Summary

```
MONTH 1 — MAKE IT WORK
  Week 1:  Brain transplant (Claude API replaces broken ensemble)
  Week 2:  OCR fix (all frames, correct pipeline order)
  Week 3:  Translation + multi-language support
  Week 4:  Spotify quality + TMDb enrichment verification

MONTH 2 — MAKE IT RIGHT
  Week 5:  Frame timeline UI + visual provenance
  Week 6:  Category-specific output components (music, listicle, recipe, shopping)
  Week 7:  Error handling + reliability (retries, fallbacks, partial results)
  Week 8:  Docker + CI/CD

MONTH 3 — MAKE IT COMPLETE
  Week 9:  Recipe extraction (ingredient parser + unit normalization)
  Week 10: Recipe frontend (serving scaler, checkboxes)
  Week 11: Admin labeling tool (building training data)
  Week 12: Analytics dashboard (measuring quality)

MONTH 4 — MAKE IT RELIABLE
  Week 13: Structured logging + dead letter queue
  Week 14: Production hardening (connection pools, rate limits)
  Week 15: Parallel processing + result caching
  Week 16: Performance benchmarking

MONTH 5-6 — MAKE IT SMART (TIER 2)
  Month 5: Export training data + prepare datasets (500+ samples)
  Month 6: Train DistilBERT + EfficientNet → deploy Tier 2 classifier
            70% cost reduction. 80% of traffic uses custom models.

MONTH 7-12 — MAKE IT YOURS (TIER 3)
  Month 7-8:  Custom extraction pattern library from real data
  Month 9-10: MLOps (model registry, A/B testing, drift monitoring)
  Month 11-12: Enterprise features (multi-store, sharing, developer API)
```

---

*Last updated: March 2026. When ready to begin, start with Month 1 Week 1 and do not skip ahead.*
