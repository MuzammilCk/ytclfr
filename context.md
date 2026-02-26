# 🧠 AI CONTEXT FILE — YouTube Intelligent Classifier
# Drop this file in your project root as `CONTEXT.md` or `.cursorrules`
# This gives your AI assistant full understanding of the codebase, architecture, bugs, and fixes.

---

## PROJECT IDENTITY

- **Name:** YouTube Intelligent Classifier (ytclfr)
- **Type:** Final Year Project — Computer Science & Engineering
- **Goal:** AI-powered YouTube video analysis — multi-modal classification using computer vision, speech-to-text, and NLP, with structured data extraction and shopping enrichment
- **Status:** Core implementation complete. Has known bugs that are documented and fixed below.
- **Target completion:** August 2026

---

## TECH STACK

### Backend
| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI 0.111 (async) |
| Task Queue | Celery + RabbitMQ broker |
| ORM | SQLAlchemy (async) + Alembic migrations |
| Primary DB | PostgreSQL 16 — job metadata |
| Document DB | MongoDB 7 — full analysis results |
| Cache / Broker | Redis 7 |
| Video Download | yt-dlp |
| Frame Extraction | OpenCV |
| Transcription | Whisper (faster-whisper) |
| Object Detection | YOLOv8 (ultralytics) |
| OCR | pytesseract + Tesseract system binary |
| NLP | spaCy 3.8.2 + en_core_web_sm |
| Vision Classifier | EfficientNet (torchvision pretrained) |
| Text Classifier | BERT/DistilBERT (transformers) |
| Auth | JWT (python-jose) |
| Settings | pydantic-settings |

### Frontend
| Layer | Technology |
|-------|-----------|
| Framework | React 18 |
| Build Tool | Vite |
| API Communication | fetch + polling |

### External APIs
| Service | Purpose | Auth |
|---------|---------|------|
| Spotify Web API | Track search, playlist generation | Client ID + Secret (OAuth2) |
| TMDb API | Movie/TV ratings, streaming availability | API Key (v3) |

---

## DIRECTORY STRUCTURE

```
youtube-classifier/                  ← project root
├── backend/
│   ├── main.py                      ← FastAPI app entrypoint, registers all routers
│   ├── core/
│   │   └── config.py                ← pydantic-settings: reads all env vars
│   ├── db/
│   │   ├── database.py              ← async SQLAlchemy engine + MongoDB + Redis clients
│   │   └── models.py                ← SQLAlchemy ORM models (Analysis table)
│   ├── models/
│   │   └── schemas.py               ← Pydantic request/response schemas
│   ├── api/
│   │   ├── routes/
│   │   │   ├── analysis.py          ← /api/v1/analyses/* endpoints
│   │   │   ├── analytics.py         ← /api/v1/analytics/* endpoints
│   │   │   └── auth.py              ← /api/v1/auth/register + login
│   │   └── middleware/
│   │       └── rate_limiter.py      ← per-IP rate limiting middleware
│   ├── services/
│   │   ├── pipeline.py              ← Celery task orchestrator (main analysis flow)
│   │   ├── video_processor/
│   │   │   ├── downloader.py        ← yt-dlp wrapper → saves mp4 to /tmp
│   │   │   └── frame_extractor.py   ← OpenCV frame sampler
│   │   ├── audio_processor/
│   │   │   └── transcriber.py       ← faster-whisper STT → timestamped transcript
│   │   ├── classification/
│   │   │   └── classifier.py        ← EfficientNet + BERT ensemble classifier
│   │   ├── extraction/
│   │   │   └── extractors.py        ← category-specific structured data extractors
│   │   ├── vision/
│   │   │   ├── yolo_detector.py     ← YOLOv8 object detection on frames
│   │   │   └── ocr_service.py       ← pytesseract OCR on frames
│   │   └── integration/
│   │       ├── spotify_service.py   ← Spotify search + playlist creation
│   │       └── tmdb_service.py      ← TMDb movie/TV lookup + JustWatch data
│   ├── alembic/
│   │   ├── env.py                   ← Alembic migration environment
│   │   └── versions/
│   │       └── 0001_initial_schema.py ← initial DB schema migration
│   └── requirements.txt             ← Python dependencies (HAS BUGS — see below)
├── frontend/
│   ├── src/
│   │   ├── App.jsx                  ← main React app: URL input, status polling, results display
│   │   └── main.jsx                 ← React entrypoint
│   ├── index.html
│   └── vite.config.js
├── scripts/
│   ├── download_models.py           ← downloads Whisper + YOLOv8 weights
│   ├── train_frame_classifier.py    ← fine-tune EfficientNet
│   ├── train_text_classifier.py     ← fine-tune BERT
│   └── prepare_dataset.py           ← dataset prep helper
├── tests/
│   ├── unit/
│   │   ├── test_extraction.py
│   │   ├── test_shopping_extractor.py
│   │   ├── test_yolo_detector.py
│   │   └── test_config_settings.py
│   ├── integration/
│   │   ├── test_api.py
│   │   ├── test_analytics.py
│   │   └── test_auth.py
│   └── conftest.py
├── .env                             ← local secrets (never commit)
├── .env.example                     ← template for .env
├── pyproject.toml                   ← pytest config (MUST EXIST — see bugs)
├── CONTEXT.md                       ← this file
└── README.md
```

---

## SYSTEM ARCHITECTURE & DATA FLOW

```
Browser
  │
  │  POST /api/v1/analyses/  { url: "youtube.com/..." }
  ▼
FastAPI (main.py)
  │  Creates DB row (status=queued), dispatches Celery task
  │
  ▼
Celery Worker (pipeline.py) ← RabbitMQ broker
  │
  ├─1─ yt-dlp          → download video to /tmp/ytclassifier/{job_id}/video.mp4
  ├─2─ OpenCV           → extract frames every N seconds
  ├─3─ faster-whisper   → transcribe audio → timestamped transcript
  ├─4─ EfficientNet     → classify frames   ┐
  │    BERT             → classify text     ┘ ensemble → category
  ├─5─ Extractor        → category-specific structured data
  │    (shopping/music/listicle/edu/news/gaming/vlog)
  ├─6─ YOLO + OCR       → detect objects + extract text from frames
  ├─7─ Spotify / TMDb   → enrich with external data
  └─8─ Save results
       ├── PostgreSQL  → status=completed, category
       └── MongoDB     → full structured result document

Browser polls GET /api/v1/analyses/{id}/status every 2s
  → on completed: GET /api/v1/analyses/{id}/result
```

---

## API ENDPOINTS

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/auth/register` | None | Create user account |
| POST | `/api/v1/auth/login` | None | Get JWT token |
| POST | `/api/v1/analyses/` | JWT | Submit YouTube URL for analysis |
| GET | `/api/v1/analyses/{id}/status` | JWT | Poll job status |
| GET | `/api/v1/analyses/{id}/result` | JWT | Get full analysis result |
| POST | `/api/v1/analyses/export` | JWT | Export result (json/csv/pdf) |
| POST | `/api/v1/analyses/batch` | JWT | Submit multiple URLs |
| GET | `/api/v1/analytics/` | JWT | Dashboard metrics |

---

## ENVIRONMENT VARIABLES

All variables are read by `backend/core/config.py` using pydantic-settings.

```dotenv
# REQUIRED — must be set before running
SECRET_KEY=                    # JWT signing key. Generate: python -c "import secrets; print(secrets.token_hex(32))"

# PostgreSQL
POSTGRES_USER=postgres
POSTGRES_PASSWORD=             # set this
POSTGRES_DB=ytclassifier
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Redis
REDIS_URL=redis://localhost:6379/0

# MongoDB
MONGO_URL=mongodb://localhost:27017

# ML
WHISPER_MODEL_SIZE=base        # tiny/base/small/medium/large
TORCH_DEVICE=cpu               # cpu or cuda
WHISPER_DEVICE=cpu             # cpu or cuda

# Optional — enables Spotify track/playlist enrichment
SPOTIFY_CLIENT_ID=
SPOTIFY_CLIENT_SECRET=

# Optional — enables TMDb movie/TV ratings enrichment
TMDB_API_KEY=
```

---

## VIDEO CATEGORIES & OUTPUTS

| Category | Extracted Data |
|----------|---------------|
| 🎭 Comedy/Entertainment | Timestamped punchlines, sentiment arc, full transcript |
| 📋 Listicle/Ranking | Ranked items + TMDb ratings + streaming availability |
| 🎵 Music Compilation | Track list + Spotify links + auto-generated playlist |
| 🎓 Educational/Tutorial | Chapters, key concepts, step-by-step breakdown |
| 📰 News/Documentary | Named entities, key points, summary |
| ⭐ Product Review | Key claims, pros/cons, shopping links (Amazon/eBay) |
| 🎮 Gaming/Esports | Game titles, players, key moments |
| 📹 Vlog/Lifestyle | Topics, locations, people mentioned |

---

## ⚠️ ALL KNOWN BUGS & EXACT FIXES

### BUG 1 — `backend/requirements.txt` — spaCy GitHub URL (BREAKS pip install on proxied networks)

**File:** `backend/requirements.txt`

**What's broken:**
```
# This line causes install failure on proxied/enterprise networks:
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

**Fix — replace with:**
```
spacy==3.8.2
en-core-web-sm==3.8.0
```

**Also run after pip install:**
```bash
python -m spacy download en_core_web_sm
```

---

### BUG 2 — `backend/requirements.txt` — pymongo version doesn't exist on PyPI

**File:** `backend/requirements.txt`

**What's broken:**
```
pymongo==4.16.0
```
This version does not exist. pip install will fail with "No matching distribution found".

**Fix:**
```
pymongo==4.11.1
```

---

### BUG 3 — `pyproject.toml` missing — pytest can't find backend modules

**File:** `pyproject.toml` (project root — may not exist)

**What's broken:**
pytest fails at collection with `ModuleNotFoundError` because `backend/` is not on sys.path.
Tests use `from main import app` and `from services.pipeline import ...` which only work if `backend/` is on the Python path.

**Fix — create/update `pyproject.toml` in project root:**
```toml
[tool.pytest.ini_options]
pythonpath = ["backend"]
asyncio_default_fixture_loop_scope = "function"
```

---

### BUG 4 — `backend/services/vision/yolo_detector.py` — bbox parsing crash

**File:** `backend/services/vision/yolo_detector.py`

**What's broken:**
```python
bbox = detection.boxes.xyxy[0].tolist()
# Crashes with AttributeError when input is already a plain Python list (e.g. in unit tests with mocks)
```

**Fix:**
```python
raw = detection.boxes.xyxy[0]
bbox = raw.tolist() if hasattr(raw, "tolist") else list(raw)
```

---

### BUG 5 — `backend/main.py` — missing CORS middleware (frontend gets blocked)

**File:** `backend/main.py`

**What's broken:**
Browser requests from `localhost:5173` (Vite dev server) are blocked by CORS policy because no CORS headers are set.

**Fix — add this block after `app = FastAPI(...)` in main.py:**
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

### BUG 6 — `backend/db/database.py` — sync SQLAlchemy engine used with async FastAPI routes

**File:** `backend/db/database.py`

**What's broken:**
If the engine is created with `create_engine` (sync) instead of `create_async_engine`, FastAPI async route handlers will throw `greenlet_spawn` errors at runtime.

**Fix — ensure async engine:**
```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# Note: replace postgresql:// with postgresql+asyncpg://
engine = create_async_engine(
    settings.POSTGRES_URL.replace("postgresql://", "postgresql+asyncpg://"),
    echo=False,
)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
```

Also ensure `asyncpg` is in requirements.txt:
```
asyncpg==0.29.0
```

---

### BUG 7 — `frontend/.env` missing — API calls go to wrong URL

**File:** `frontend/.env` (create this file — it does not exist by default)

**What's broken:**
Without this file, the frontend has no API base URL and fetch calls fail or go to the wrong host.

**Fix — create `frontend/.env`:**
```
VITE_API_URL=http://localhost:8000
```

**Then in `frontend/src/App.jsx`, ensure API base URL uses:**
```javascript
const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
```

---

### BUG 8 — Celery worker crashes on Windows (Windows-only)

**What's broken:**
On Windows, Celery's default `prefork` worker pool uses multiprocessing which is not supported. Worker exits immediately with no useful error.

**Fix — always use `--pool=solo` on Windows:**
```powershell
celery -A services.pipeline.celery_app worker --pool=solo --loglevel=info
```

---

### BUG 9 — Tesseract OCR binary missing (system dependency)

**File:** `backend/services/vision/ocr_service.py`

**What's broken:**
`pytesseract` is a Python wrapper around the Tesseract binary. If the binary isn't installed and on PATH, every OCR call throws `TesseractNotFoundError`.

**Fix:**
- **Windows:** Download from https://github.com/UB-Mannheim/tesseract/wiki → install → add install dir to PATH
- **Ubuntu/Debian:** `sudo apt-get install tesseract-ocr`
- **Mac:** `brew install tesseract`

**Verify:** `tesseract --version`

---

## BUG FIX STATUS SUMMARY

| # | File | Bug | Status |
|---|------|-----|--------|
| 1 | `backend/requirements.txt` | spaCy GitHub URL breaks pip | ✅ Fix documented |
| 2 | `backend/requirements.txt` | `pymongo==4.16.0` doesn't exist | ✅ Fix documented |
| 3 | `pyproject.toml` | pytest ModuleNotFoundError | ✅ Fix documented |
| 4 | `services/vision/yolo_detector.py` | bbox `.tolist()` crash | ✅ Fix documented |
| 5 | `backend/main.py` | No CORS headers | ✅ Fix documented |
| 6 | `backend/db/database.py` | Sync engine with async routes | ✅ Fix documented |
| 7 | `frontend/.env` | Missing VITE_API_URL | ✅ Fix documented |
| 8 | Celery (Windows) | Worker crashes without --pool=solo | ✅ Fix documented |
| 9 | System | Tesseract binary not installed | ✅ Fix documented |

---

## PIPELINE DETAILED FLOW

```
pipeline.py  →  Celery task: analyze_video(job_id, url)
│
├── Step 1: downloader.py
│     yt_dlp.YoutubeDL → download best mp4 quality
│     saves to: /tmp/ytclassifier/{job_id}/video.mp4
│     updates PostgreSQL: status = "downloading"
│
├── Step 2: frame_extractor.py
│     cv2.VideoCapture → sample 1 frame per N seconds
│     saves frames as: /tmp/ytclassifier/{job_id}/frames/frame_{n}.jpg
│     updates PostgreSQL: status = "extracting_frames"
│
├── Step 3: transcriber.py
│     WhisperModel.transcribe() → list of segments with timestamps
│     returns: [{ start, end, text }, ...]
│     updates PostgreSQL: status = "transcribing"
│
├── Step 4: classifier.py
│     EfficientNet → softmax over frames → top category per frame → vote
│     BERT → classify transcript text → category + confidence
│     Ensemble: weighted average (configurable, default 60% text / 40% vision)
│     returns: { category: str, confidence: float }
│     updates PostgreSQL: status = "classifying", category = result
│
├── Step 5: extractors.py
│     Routes to category-specific extractor:
│     ShoppingExtractor  → uses YOLO + OCR + spaCy NER
│     MusicExtractor     → parses transcript for track/artist patterns
│     ListicleExtractor  → extracts ranked items
│     EducationalExtractor → extracts chapters/concepts
│     NewsExtractor      → named entities, key points
│     GamingExtractor    → game titles, player names
│     VlogExtractor      → topics, locations, people
│     updates PostgreSQL: status = "extracting"
│
├── Step 6: yolo_detector.py + ocr_service.py (used inside extractors)
│     YOLO → bounding boxes + class labels for objects in frames
│     pytesseract → extract text/prices from frame regions
│
├── Step 7: spotify_service.py / tmdb_service.py
│     Spotify: search tracks → get URIs → create playlist
│     TMDb: search movies/shows → get ratings + streaming availability
│     updates PostgreSQL: status = "enriching"
│
└── Step 8: save results
      MongoDB: insert full result document
      PostgreSQL: status = "completed"
      Redis: cache result for fast subsequent GETs
      cleanup: shutil.rmtree(/tmp/ytclassifier/{job_id})
```

---

## DATABASE SCHEMAS

### PostgreSQL — `analyses` table (models.py)
```python
id            UUID primary key
url           String — original YouTube URL
status        Enum: queued / processing / downloading / extracting_frames /
              transcribing / classifying / extracting / enriching / completed / failed
category      String nullable — detected category
celery_task_id String nullable
error_message String nullable
created_at    DateTime
updated_at    DateTime
user_id       UUID foreign key → users table
```

### MongoDB — analysis result document
```json
{
  "analysis_id": "uuid",
  "category": "shopping",
  "confidence": 0.91,
  "transcript": [{ "start": 0.0, "end": 3.2, "text": "..." }],
  "extracted": {
    "items": [
      {
        "name": "Product Name",
        "confidence": 0.85,
        "timestamp": 42.5,
        "search_links": {
          "amazon": "https://amazon.com/s?k=Product+Name",
          "ebay": "https://ebay.com/sch/i.html?_nkw=Product+Name"
        }
      }
    ]
  },
  "enrichment": { ... }
}
```

---

## CLASSIFICATION SYSTEM

### How the ensemble works
```
Frames (video) → EfficientNet (pretrained ImageNet)
                    → predict category per frame
                    → majority vote → frame_category, frame_confidence

Transcript (text) → DistilBERT (pretrained)
                    → predict category for full text
                    → text_category, text_confidence

Ensemble:
  final_confidence = (0.6 × text_confidence) + (0.4 × frame_confidence)
  final_category = highest combined score
```

### Zero-shot baseline accuracy: ~70%
### Fine-tuned accuracy target: ≥85%

### Fine-tuning (optional):
```bash
python scripts/train_frame_classifier.py --epochs 20
python scripts/train_text_classifier.py --epochs 10
# Weights saved to /tmp/ytclassifier/models/
```

---

## SHOPPING EXTRACTION DETAIL

The shopping extractor (used when category = "product_review" or "shopping") combines 3 signals:

```
Signal 1 — YOLO object detection
  yolo_detector.py → YOLOv8n model
  detects: common product classes in frames
  returns: [{ class_name, confidence, bbox, frame_timestamp }]

Signal 2 — OCR text extraction
  ocr_service.py → pytesseract on frame regions
  extracts: product names, prices, brand text visible on screen

Signal 3 — spaCy NER on transcript
  extractors.py → spaCy en_core_web_sm
  extracts: PRODUCT, ORG, GPE entities from speech

Combined → deduplicated product list
         → shopping search links generated:
           Amazon:  https://www.amazon.com/s?k={urllib.parse.quote(name)}
           eBay:    https://www.ebay.com/sch/i.html?_nkw={urllib.parse.quote(name)}
```

---

## FRONTEND COMPONENT FLOW

```
App.jsx
│
├── State: { token, user, url, jobId, status, result, view }
│
├── Auth views
│   ├── LoginForm  → POST /api/v1/auth/login  → stores JWT in state
│   └── RegisterForm → POST /api/v1/auth/register
│
├── Main view (after login)
│   ├── URLInput + Analyse button
│   │   └── POST /api/v1/analyses/ → sets jobId
│   │
│   ├── StatusPoller (runs every 2s while status != completed/failed)
│   │   └── GET /api/v1/analyses/{jobId}/status
│   │
│   └── ResultDisplay (shown when status = completed)
│       ├── Tab: Output     — category-specific structured data
│       ├── Tab: Classification — category + confidence score
│       └── Tab: Transcript — full timestamped transcript
│
└── Analytics view (📊 button)
    └── GET /api/v1/analytics/ → charts + metrics dashboard
```

---

## STARTUP ORDER (Windows, no Docker)

```
1. Start PostgreSQL service (Windows Services or pg_ctl)
2. Start Memurai/Redis service (Windows Services)
3. Start MongoDB service (Windows Services or mongod)
4. Terminal 1 → backend API:
     cd backend && .venv\Scripts\Activate.ps1
     uvicorn main:app --reload --host 0.0.0.0 --port 8000
5. Terminal 2 → Celery worker:
     cd backend && .venv\Scripts\Activate.ps1
     celery -A services.pipeline.celery_app worker --pool=solo --loglevel=info
6. Terminal 3 → Frontend:
     cd frontend && npm run dev
7. Open http://localhost:5173
```

---

## WHAT IS NOT YET IMPLEMENTED (production gaps)

These features are in the vision doc but NOT in the current codebase:

1. **Live price comparison** — no real Amazon/eBay/Walmart API adapters. Currently only generates search URLs, does not fetch live prices.
2. **Dead letter queue / retry policy** — Celery tasks have no retry logic. Failed jobs stay failed.
3. **Distributed tracing** — no OpenTelemetry / Jaeger integration.
4. **Model versioning** — no MLflow or model registry. Models loaded from fixed paths.
5. **Secret rotation** — secrets are static env vars, no AWS Secrets Manager integration.
6. **PII data governance** — no data retention policy, deletion workflows, or GDPR compliance layer.
7. **Affiliate attribution** — shopping links are plain search URLs with no affiliate tracking.
8. **GPU autoscaling** — no dynamic worker scaling based on queue depth.

---

## TESTING

```bash
# Unit tests — no external services needed
cd backend
pytest tests/unit/ -v

# Integration tests — requires running Postgres + Redis + MongoDB
pytest tests/integration/ -v

# Full suite with HTML coverage report
pytest --cov=. --cov-report=html

# Coverage target: ≥ 90%
```

**Test files and what they test:**
- `test_extraction.py` — category-specific extractors
- `test_shopping_extractor.py` — shopping item extraction + link generation
- `test_yolo_detector.py` — YOLO bbox parsing (includes mock/list input edge case — Bug 4)
- `test_config_settings.py` — pydantic-settings env var loading
- `test_api.py` — full HTTP integration tests for analysis endpoints
- `test_analytics.py` — analytics endpoint integration tests
- `test_auth.py` — register/login/JWT integration tests

---

## KEY RULES FOR AI ASSISTANCE

When making changes to this project, always follow these rules:

1. **Never use sync SQLAlchemy** in async FastAPI routes — always use `AsyncSession` and `create_async_engine` with `asyncpg` driver.
2. **Never start Celery on Windows** without `--pool=solo`.
3. **Never add GitHub wheel URLs** to requirements.txt — always use PyPI package names with pinned versions.
4. **Always check for `hasattr(x, "tolist")`** before calling `.tolist()` on YOLO detection output.
5. **Always add CORS middleware** to main.py when running frontend on a different port.
6. **pyproject.toml must exist** at project root with `pythonpath = ["backend"]` for tests to work.
7. **frontend/.env must exist** with `VITE_API_URL=http://localhost:8000` for API calls to work.
8. **Temp files must be cleaned up** after each Celery task — use `shutil.rmtree` on `/tmp/ytclassifier/{job_id}`.
9. **MongoDB stores full results** — PostgreSQL only stores job metadata and status. Never store large result payloads in PostgreSQL.
10. **All FastAPI route files** are in `backend/api/routes/` and must be registered in `backend/main.py` with `app.include_router(...)`.
