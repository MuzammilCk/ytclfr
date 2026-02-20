# Remaining Work — YouTube Intelligent Classifier

Track all outstanding implementation items here.
Status: `[ ]` todo · `[/]` in-progress · `[x]` done

---

## Phase 2 Leftover — PDF Export Layouts

- [x] **`backend/api/routes/analysis.py`** — Structured PDF sections for `music` (Rank/Title/Artist/Year/Spotify URL table) and `educational` (chapter table + key-concepts + summary) added. Also added `shopping` branch and a generic key/value fallback. Listicle header styling improved too.

---

## Phase 4 — Shopping / YOLO Feature

- [x] **`backend/services/vision/yolo_detector.py`** *(NEW)* — `YOLODetector` wrapping `ultralytics.YOLO`; `detect(frame_paths) -> List[Detection]`; thread-pool executor; graceful degradation if model not loaded.
- [x] **`backend/requirements.txt`** — `ultralytics==8.2.18` already present.
- [x] **`backend/services/extraction/extractors.py`** — `ShoppingExtractor(BaseExtractor)` added; YOLO detections aggregated + deduped; NLP `PRODUCT` entities merged; Google Shopping search URLs; `shopping` entry in factory map.
- [x] **`backend/db/models.py`** — `SHOPPING = "shopping"` added to `VideoCategory` enum.
- [ ] **`backend/alembic/versions/`** — New migration needed: `alembic revision --autogenerate -m "add_shopping_category"` then `alembic upgrade head`. *(Run manually after setting up local Postgres.)*
- [x] **`backend/models/schemas.py`** — `ProductItem` and `ShoppingOutput` Pydantic models added; `ShoppingOutput` added to `AnalysisResult.output` union.
- [x] **`backend/services/pipeline.py`** — `YOLODetector` singleton imported; YOLO detection runs for shopping category before extraction; detections injected into `ShoppingExtractor`.
- [x] **`backend/api/routes/analysis.py`** — PDF export: `shopping` branch with product table. CSV export: `shopping` branch with Name/Brand/Category/Confidence/Search URL.
- [x] **`frontend/src/App.jsx`** — `ShoppingOutput` React component (product cards grid, confidence bar, category tags, Google Shopping links); `"shopping"` case added to `renderOutput()`.
- [x] **`tests/unit/test_yolo_detector.py`** *(NEW)* — Mocks `ultralytics.YOLO`; tests detection parsing, empty inputs, error handling, `is_available()`.
- [x] **`tests/unit/test_shopping_extractor.py`** *(NEW)* — Tests `ShoppingExtractor.extract()`: type field, YOLO product, non-product exclusion, deduplication, URL format, category inference, zero detections.

---

## Phase 5 — DevOps

- [x] **`backend/services/pipeline.py`** — Already reads `CELERY_BROKER_URL` from `settings`. Confirmed, no change needed.
- [x] **`SETUP.md`** *(NEW, project root)* — Windows no-Docker local setup guide: PostgreSQL, Redis (Memurai), MongoDB, model downloads, Alembic migrations, Celery worker, uvicorn, frontend dev server, env variable reference, troubleshooting table.
- [x] **`scripts/download_models.py`** *(NEW)* — Downloads Whisper `base` and `yolov8n.pt`; prints file sizes; smoke-tests YOLO; supports `--whisper`/`--yolo` flags + env var overrides; exits non-zero on failure.
- [x] **`.github/workflows/ci.yml`** *(NEW)* — GitHub Actions: Python 3.11+3.12 matrix; `services:` block for Postgres 16, Redis 7, MongoDB 7; Alembic migrations; `pytest tests/unit/` + `pytest tests/integration/`; `npm run build` frontend job; uploads JUnit XML artifacts.

---

## One remaining manual step

> **Alembic migration for `shopping` enum value** — after pulling these changes, run:
> ```powershell
> cd backend
> alembic revision --autogenerate -m "add_shopping_category"
> alembic upgrade head
> ```
> This alters the `videocategory` Postgres enum to include `shopping`.
