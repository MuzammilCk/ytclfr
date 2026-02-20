# Local Setup Guide — Windows (No Docker)

This guide runs the full YouTube Intelligent Classifier stack natively on Windows without Docker.
Follow each section in order.

---

## Prerequisites

| Requirement | Version | Download |
|---|---|---|
| Python | 3.11 or 3.12 | [python.org](https://www.python.org/downloads/) |
| Node.js | 20 LTS | [nodejs.org](https://nodejs.org/) |
| PostgreSQL | 16 | [postgresql.org](https://www.postgresql.org/download/windows/) |
| Redis (Memurai) | 4+ | [memurai.com](https://www.memurai.com/) — Windows-native Redis fork |
| MongoDB Community | 7 | [mongodb.com](https://www.mongodb.com/try/download/community) |
| Git | latest | [git-scm.com](https://git-scm.com/) |
| FFmpeg | 6+ | [ffmpeg.org](https://ffmpeg.org/download.html) — add to PATH |

> **GPU (optional):** If you have an NVIDIA GPU, install [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads) and set `TORCH_DEVICE=cuda` and `WHISPER_DEVICE=cuda` in your `.env`.

---

## 1. Clone and configure environment

```powershell
git clone <your-repo-url>
cd youtube-classifier
```

Copy the example env file and fill in your secrets:

```powershell
Copy-Item .env.example .env
```

Open `.env` and set **at minimum**:

```dotenv
POSTGRES_PASSWORD=<your-pg-password>
SECRET_KEY=<run: python -c "import secrets; print(secrets.token_hex(32))">
```

Leave all other values as their defaults for local development.

---

## 2. Start local services

### PostgreSQL
Start the service from Windows Services, or:
```powershell
pg_ctl -D "C:\Program Files\PostgreSQL\16\data" start
```

Create the database:
```powershell
psql -U postgres -c "CREATE DATABASE ytclassifier;"
```

### Redis (Memurai)
After installing Memurai, start it from Windows Services.
Verify: `redis-cli ping` → should return `PONG`.

### MongoDB
Start via Windows Services, or:
```powershell
mongod --dbpath "C:\data\db"
```

---

## 3. Install Python dependencies

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Download spaCy English model (recommended for best extraction quality)
python -m spacy download en_core_web_sm
```

---

## 4. Download ML models

```powershell
python ..\scripts\download_models.py
```

This downloads:
- **Whisper `base`** (~145 MB) → `%USERPROFILE%\.cache\whisper\`
- **YOLOv8n** (~6 MB) → `yolov8n.pt` in current directory

To download a larger Whisper model for better accuracy:
```powershell
python ..\scripts\download_models.py --whisper --whisper-size small
```

---

## 5. Run database migrations

```powershell
# Still in backend/ with .venv activated
alembic upgrade head
```

Expected output: `Running upgrade ... -> <hash>, initial schema`

---

## 6. Start the backend

**Terminal 1 — FastAPI server:**
```powershell
cd backend
.venv\Scripts\Activate.ps1
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
API docs: http://localhost:8000/api/docs

**Terminal 2 — Celery worker:**
```powershell
cd backend
.venv\Scripts\Activate.ps1
celery -A services.pipeline.celery_app worker --loglevel=info --concurrency=2
```

> On Windows, Celery requires the `solo` pool for proper operation:
> `celery -A services.pipeline.celery_app worker --pool=solo --loglevel=info`

---

## 7. Start the frontend

```powershell
cd frontend
npm install
npm run dev
```

Open: http://localhost:5173

---

## 8. Verify everything works

1. Open http://localhost:5173
2. Register a new account via the **Log in** button → Register tab
3. Paste a YouTube URL (e.g. `https://www.youtube.com/watch?v=dQw4w9WgXcQ`)
4. Click **Analyse** — the pipeline should progress through all status steps
5. Results appear with the Output / Classification / Transcript tabs
6. Click **📊 Analytics** to verify the dashboard data loads

---

## Environment variables reference

| Variable | Purpose | Default |
|---|---|---|
| `SECRET_KEY` | JWT signing key | **must set** |
| `POSTGRES_*` | Database connection | localhost / ytclassifier |
| `REDIS_URL` | Cache & Celery broker | `redis://localhost:6379/0` |
| `WHISPER_MODEL_SIZE` | `tiny` / `base` / `small` / `medium` / `large` | `base` |
| `TORCH_DEVICE` | `cpu` or `cuda` | `cpu` |
| `SPOTIFY_CLIENT_ID/SECRET` | Music enrichment (optional) | empty |
| `TMDB_API_KEY` | Movie/TV enrichment (optional) | empty |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `alembic upgrade head` fails | Check `POSTGRES_*` in `.env`; verify database exists |
| Celery worker exits immediately | Use `--pool=solo` on Windows |
| Whisper runtime error | Verify `faster-whisper` installed and FFmpeg is in PATH; then re-run `download_models.py` |
| Frontend can't reach API | Set `VITE_API_URL=http://localhost:8000` in `frontend/.env` |
| `en_core_web_sm` not found | Run `python -m spacy download en_core_web_sm` (or install from your internal package mirror) |
