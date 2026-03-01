# Local Setup Guide — Enterprise Deployment (Windows)

This document outlines the requisite steps to configure, deploy, and execute the **YouTube Intelligent Classifier** stack natively on a Windows host machine. The guide aligns with enterprise-level developer onboarding standards.

Please execute each phase sequentially. Failure to properly initialize dependencies (e.g., database instances, environment variables) will result in application runtime exceptions.

---

## 🛑 Prerequisites & Dependencies

Ensure the host machine is provisioned with the following system software before progressing. Administrative privileges are recommended for service installations.

| Software Requirement | Version Constraint | Reference / Download Link |
|---|---|---|
| **Python** | `3.11` or `3.12` | [Download Python](https://www.python.org/downloads/) |
| **Node.js** | `v20 LTS` | [Download Node.js](https://nodejs.org/) |
| **PostgreSQL** | `v16.x` | [Download PostgreSQL](https://www.postgresql.org/download/windows/) |
| **MongoDB Community**| `v7.x` | [Download MongoDB](https://www.mongodb.com/try/download/community) |
| **Redis Server** | `v4.x` / Memurai | [Download Memurai (Windows Redis)](https://www.memurai.com/) |
| **FFmpeg** | `v6.x+` | [Download FFmpeg](https://ffmpeg.org/download.html) (Must set in `PATH`) |
| **Tesseract OCR** | Latest | [Tesseract Installer](https://github.com/UB-Mannheim/tesseract/wiki) (Must set in `PATH`) |
| **Git CLI** | Latest | [Download Git](https://git-scm.com/) |

> **Hardware Acceleration (Optional but Recommended)**:
> If an NVIDIA GPU is present, allocate hardware inference acceleration by mapping `TORCH_DEVICE=cuda` and `WHISPER_DEVICE=cuda`. You must pre-install the [NVIDIA CUDA Toolkit (v12.x)](https://developer.nvidia.com/cuda-downloads).

---

## 🔐 Phase 1: Environment & Secrets Configuration

1. **Clone the Source Repository:**
   ```powershell
   git clone <repository-url>
   cd youtube-classifier
   ```

2. **Establish the Backend Environment Manifest (`.env`):**
   ```powershell
   Copy-Item .env.example .env
   ```
   Open `.env` and configure your credentials. At minimum, secure the following:
   ```dotenv
   # Securely formulate a 32-byte hexadecimal key:
   # python -c "import secrets; print(secrets.token_hex(32))"
   SECRET_KEY=your_generated_secret_key_here
   
   # Database Credentials
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=your_secure_db_password
   
   # ML Optimization Parameters
   WHISPER_MODEL_SIZE=base
   TORCH_DEVICE=cpu
   ```

3. **Establish the Frontend Configuration:**
   Create a dedicated `.env` inside the `frontend/` directory to orient API requests:
   ```powershell
   # In youtube-classifier/frontend/.env
   VITE_API_URL=http://localhost:8000
   ```

---

## 🛠️ Phase 2: Service & Infrastructure Instantiation

Ensure your local caching, messaging, and storage solutions are actively running. 

### 1. Relational Database (PostgreSQL)
Start the PostgreSQL server via Windows Services (`services.msc`), or execute via CLI targeting the data path:
```powershell
pg_ctl -D "C:\Program Files\PostgreSQL\16\data" start
```
Initialize the application database:
```powershell
psql -U postgres -c "CREATE DATABASE ytclassifier;"
```

### 2. Document Store (MongoDB)
Start the MongoDB daemon targeting the data layer:
```powershell
mongod --dbpath "C:\data\db"
```

### 3. Key-Value Broker (Memurai / Redis)
Validate that Memurai is executing natively in Windows Services. Validate operability via command line:
```powershell
redis-cli ping
# Expected Standard Out: PONG
```

---

## 📦 Phase 3: Python Environment & Migrations

Isolate your Python dependencies using virtual environments, followed by installing deterministic application libraries.

### 1. Backend Initialization
```powershell
# Navigate from project root
cd backend

# Create & Source the Virtual Environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install strict PIP packages
pip install -r requirements.txt

# Hydrate spaCy linguistic components globally
python -m spacy download en_core_web_sm
```

### 2. ML Models Artifact Recovery
Pre-fetch machine learning models (Whisper ASR, YOLOv8) explicitly rather than lazy-loading them during job execution:
```powershell
# Assuming current dir is youtube-classifier/backend/
python ..\scripts\download_models.py
```
*(Optionally append `--whisper-size large` if adequate GPU VRAM is available space).*

### 3. Database Schema Ascension
Migrate the async PostgreSQL schema automatically via Alembic:
```powershell
# Ensure the backend virtual environment is active
alembic upgrade head
```
*(Look for the standard out: `Running upgrade ... -> <hash>, initial schema`)*

---

## 🚀 Phase 4: Cluster Execution (Terminal Guidelines)

You will need **three separate powershell instances** to run the multi-tiered application architecture. Ensure background services (Postgess, Mongo, Redis) are online prior to this stage.

### Terminal 1: FastAPI Gateway
Initiate the asynchronous REST application server handling external routings.
```powershell
cd backend
.venv\Scripts\Activate.ps1
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
*Health Check*: Navigate to [http://localhost:8000/api/docs](http://localhost:8000/api/docs)

---

### Terminal 2: Celery ML Engine
Initiate the Celery orchestrator listener. 
> ⚠️ **Windows Critical Constraint**: Windows kernels reject standard prefork parallelizations natively. It is mandatory to append the `--pool=solo` flag.
```powershell
cd backend
.venv\Scripts\Activate.ps1
celery -A services.pipeline.celery_app worker --pool=solo --loglevel=info
```
*Health Check*: Expect logger notifications mentioning `[tasks]` and stating the worker process init has successfully preloaded the models into system RAM.

---

### Terminal 3: React Single Page Application
Execute the client development server.
```powershell
cd frontend
npm install
npm run dev
```
*Health Check*: Launch [http://localhost:5173](http://localhost:5173) in your browser.

---

## 🩺 Phase 5: Verification & End-to-End Walkthrough

To functionally prove the deployment behaves properly:
1. Navigate to your local React address (`http://localhost:5173`).
2. Utilize the **Authentication Form** to register an arbitrary test account profile.
3. Once authenticated locally, submit a valid YouTube Uniform Resource Locator into the query engine (e.g., `https://www.youtube.com/watch?v=dQw4w9WgXcQ`).
4. Click **Analyse**. Monitor **Terminal 2 (Celery)**. You should observe state transitions from `downloading` -> `extracting_frames` -> `transcribing` -> `classifying`.
5. Verify the generated output manifests on the UI under specific tabs (e.g., `Output`, `Classification`, `Transcript`).

---

## 🚨 Troubleshooting & Fault Isolation

| Implication / Error Signature | Root Cause Resolution |
|---|---|
| **CORS Validation Error pointing to 8000** | Confirm `frontend/.env` incorporates `VITE_API_URL=http://localhost:8000`. Stop/Start the `npm run dev` session. |
| **`alembic upgrade head` failure** | Check explicit credentials populated within the backend `.env` mapped to `POSTGRES_*`. Verify `pg_ctl` is alive. |
| **Celery worker abrupt exits** | Windows OS exception. The worker orchestrator *must* execute bearing `--pool=solo`. |
| **`TesseractNotFoundError` in Logs** | Pytesseract cannot map OS bounds to binary path. Download tesseract directly, then enforce the destination into Window's system `PATH`. Restart terminals thereafter. |
| **`ModuleNotFoundError: No module named 'main'` during `pytest`** | Ensure the `pyproject.toml` descriptor lives at the repository root and possesses `pythonpath = ["backend"]`. |
| **`TMDb` or `Spotify` fields blank** | Free API developer credentials must be generated on their respective portals and hydrated into the `.env` configs structure. |
