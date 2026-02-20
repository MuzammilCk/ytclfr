# 🎬 YouTube Intelligent Classifier

> AI-powered video analysis and structured content extraction.  
> Multi-modal classification combining computer vision, speech-to-text, and NLP.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python)](https://python.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat&logo=react)](https://react.dev)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker)](https://docker.com)

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                          FRONTEND (React + Vite)                        │
│   URL Input → Real-time Status Polling → Structured Result Display      │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │ REST API
┌──────────────────────────────▼─────────────────────────────────────────┐
│                        API GATEWAY (FastAPI)                            │
│   /api/v1/analyses  POST  →  Submit job                                 │
│   /api/v1/analyses/{id}/status  GET  →  Poll progress                  │
│   /api/v1/analyses/{id}/result  GET  →  Fetch full result               │
│   /api/v1/analyses/export  POST  →  Download JSON/CSV/PDF               │
└──────┬──────────────────────────────────────────────────────────────────┘
       │ Celery Task
┌──────▼────────────────────────────────────────────────────────────────┐
│                      ANALYSIS PIPELINE (Celery Worker)                  │
│                                                                         │
│  yt-dlp Download  →  OpenCV Frames  →  Whisper Transcription            │
│         │                  │                     │                      │
│         └──────────────────▼─────────────────────┘                      │
│                     EfficientNet + BERT Ensemble Classifier             │
│                             │                                           │
│             ┌───────────────┴──────────────┐                            │
│             │                              │                            │
│        TMDb Enrichment             Spotify Integration                  │
│    (movies, streaming info)      (track search + playlist)              │
└────────────────────────────────────────────────────────────────────────┘
       │
┌──────▼────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                      │
│   PostgreSQL (job metadata)  ·  MongoDB (full results)  ·  Redis (cache) │
└────────────────────────────────────────────────────────────────────────┘
```

## Supported Video Categories

| Category | Output |
|---|---|
| 🎭 Comedy/Entertainment | Timestamped punchlines, sentiment arc, full transcript |
| 📋 Listicle/Ranking | Ranked items with TMDb ratings & streaming availability |
| 🎵 Music Compilation | Track list with Spotify links, auto-generated playlist |
| 🎓 Educational/Tutorial | Chapters, key concepts, step-by-step breakdown |
| 📰 News/Documentary | Named entities, key points, summary |
| ⭐ Product Review | Key claims, pros/cons, entity extraction |
| 🎮 Gaming/Esports | Game titles, players, key moments |
| 📹 Vlog/Lifestyle | Topics, locations, people mentioned |

---

## Quickstart

### Prerequisites

- Docker & Docker Compose  
- Python 3.11+ (for local dev)  
- Node.js 20+ (for frontend dev)  
- 8GB+ RAM recommended (ML models)

### 1. Clone & configure

```bash
git clone https://github.com/yourname/youtube-classifier.git
cd youtube-classifier

# Copy and edit environment variables
cp .env.example .env
# Edit .env — at minimum set SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, TMDB_API_KEY
```

### 2. Start with Docker Compose

```bash
cd docker
docker compose up -d --build
```

Services will be available at:
- **Frontend**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/api/docs
- **Flower** (Celery monitor): http://localhost:5555
- **RabbitMQ UI**: http://localhost:15672

### 3. Local development (without Docker)

```bash
# Backend
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run DB migrations
alembic upgrade head

# Start API
uvicorn main:app --reload --port 8000

# In a separate terminal — start Celery worker
celery -A services.pipeline.celery_app worker --loglevel=info

# Frontend
cd frontend
npm install
npm run dev
```

---

## API Reference

### Submit a video for analysis

```http
POST /api/v1/analyses/
Content-Type: application/json

{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "force_reanalysis": false
}
```

**Response** (202 Accepted):
```json
{
  "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "estimated_seconds": 60,
  "message": "Analysis job accepted."
}
```

### Poll status

```http
GET /api/v1/analyses/{analysis_id}/status
```

### Get full result

```http
GET /api/v1/analyses/{analysis_id}/result
```

### Export result

```http
POST /api/v1/analyses/export
Content-Type: application/json

{
  "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
  "format": "csv"
}
```

Supported formats: `json`, `csv`, `pdf`

### Batch processing

```http
POST /api/v1/analyses/batch
Content-Type: application/json

{
  "urls": [
    "https://youtube.com/watch?v=video1",
    "https://youtube.com/watch?v=video2"
  ]
}
```

---

## Running Tests

```bash
# Unit tests (no external dependencies)
cd backend
pytest tests/unit/ -v

# Integration tests (requires running DB — use Docker)
pytest tests/integration/ -v

# Full test suite with coverage report
pytest --cov=. --cov-report=html
```

---

## Getting API Keys

| Service | Where to get | Cost |
|---|---|---|
| **TMDb** | https://developer.themoviedb.org | Free |
| **Spotify** | https://developer.spotify.com/dashboard | Free |

---

## Model Training

The classification system ships with ImageNet/BERT pretrained weights as a zero-shot baseline. To fine-tune on your own labelled data:

```bash
# 1. Prepare dataset: CSV with columns [video_id, url, category]
# 2. Run feature extraction
python scripts/extract_features.py --dataset data/videos.csv

# 3. Fine-tune image classifier
python scripts/train_frame_classifier.py --epochs 20

# 4. Fine-tune text classifier  
python scripts/train_text_classifier.py --epochs 10

# 5. Save weights (auto-loaded on next worker restart)
# Weights saved to /tmp/ytclassifier/models/
```

---

## Deployment to Production (AWS)

```bash
# Build and push images
docker build -t ytclassifier-api ./backend
docker build -t ytclassifier-frontend ./frontend

# Push to ECR, deploy via ECS or EKS
# See docs/deployment.md for full Terraform/Kubernetes configs
```

Key production considerations:
- Use GPU instance (g4dn.xlarge) for ML inference — 3–5× faster
- Set `WHISPER_MODEL_SIZE=medium` for better accuracy
- Use S3 for media storage instead of local `/tmp`
- Enable CloudWatch metrics via the `/metrics` Prometheus endpoint

---

## Project Structure

```
youtube-classifier/
├── backend/
│   ├── main.py                          # FastAPI app
│   ├── core/config.py                   # Settings (pydantic-settings)
│   ├── db/
│   │   ├── database.py                  # Async DB clients
│   │   └── models.py                    # SQLAlchemy ORM
│   ├── models/schemas.py                # Pydantic request/response schemas
│   ├── api/routes/analysis.py           # REST endpoints
│   ├── services/
│   │   ├── pipeline.py                  # Celery task orchestrator
│   │   ├── video_processor/
│   │   │   ├── downloader.py            # yt-dlp wrapper
│   │   │   └── frame_extractor.py       # OpenCV frame sampling
│   │   ├── audio_processor/
│   │   │   └── transcriber.py           # Whisper STT
│   │   ├── classification/
│   │   │   └── classifier.py            # EfficientNet + BERT ensemble
│   │   ├── extraction/
│   │   │   └── extractors.py            # Category-specific extractors
│   │   └── integration/
│   │       ├── spotify_service.py       # Spotify Web API
│   │       └── tmdb_service.py          # TMDb + JustWatch
│   ├── alembic/                         # DB migrations
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx                      # Main React app
│   │   └── main.jsx
│   ├── index.html
│   └── package.json
├── docker/
│   ├── docker-compose.yml
│   └── nginx.conf
├── tests/
│   ├── unit/test_extraction.py
│   └── integration/test_api.py
└── .env.example
```

---

## Success Metrics

| Metric | Target | How measured |
|---|---|---|
| Classification accuracy | ≥ 85% | Test set of 1000 labelled videos |
| Processing time (10 min video) | ≤ 3 min | P95 latency from Celery task logs |
| Transcription WER | ≤ 10% | Whisper on clean-audio benchmark |
| System uptime | ≥ 99% | Prometheus + alertmanager |
| Code coverage | ≥ 90% | pytest-cov |

---

*Final Year Project — Computer Science & Engineering*  
*Expected completion: August 2026*
# ytclfr
