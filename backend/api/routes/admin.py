"""
api/routes/admin.py

Admin-only endpoints for training data labeling and export.
All routes require admin authentication.
"""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from loguru import logger

from api.routes.auth import get_current_admin
from core.config import get_settings
from db.models import User
from models.schemas import LabelRequest, OKResponse, TrainingSampleMeta

settings = get_settings()

router = APIRouter(
    prefix="/api/v1/admin",
    tags=["Admin Training"],
    dependencies=[Depends(get_current_admin)],
)

TRAINING_DIR = Path(settings.TRAINING_DATA_DIR)


def _load_sample(sample_id: str) -> dict:
    """Load a training sample JSON by sample_id. Raises 404 if not found."""
    path = TRAINING_DIR / f"{sample_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found")
    try:
        text = path.read_text(encoding="utf-8").strip()
        return json.loads(text) if text else {}
    except json.JSONDecodeError:
        return {}


# ── 8.2: Label a sample ───────────────────────────────────────────────────────

@router.post("/label", response_model=OKResponse)
async def label_sample(
    body: LabelRequest,
    current_admin: User = Depends(get_current_admin),
):
    """Write a human_label to an existing training sample JSON."""
    sample = _load_sample(body.sample_id)
    sample["human_label"] = body.human_label
    sample["labeled_by"] = str(current_admin.id)

    path = TRAINING_DIR / f"{body.sample_id}.json"
    path.write_text(json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"[admin] Sample {body.sample_id} labeled as '{body.human_label}' by {current_admin.email}")

    return OKResponse(message=f"Sample '{body.sample_id}' labeled as '{body.human_label}'")


# ── 8.2: List all training samples ───────────────────────────────────────────

@router.get("/training-data", response_model=List[TrainingSampleMeta])
async def list_training_samples():
    """List all training samples with their metadata and labeling status."""
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for path in sorted(TRAINING_DIR.glob("*.json")):
        sample_id = path.stem
        try:
            text = path.read_text(encoding="utf-8").strip()
            data = json.loads(text) if text else {}
        except (json.JSONDecodeError, OSError):
            data = {}

        results.append(TrainingSampleMeta(
            sample_id=sample_id,
            video_title=data.get("video_title"),
            predicted_category=data.get("predicted_category"),
            confidence=data.get("confidence"),
            human_label=data.get("human_label"),
            is_labeled=data.get("human_label") is not None,
        ))

    return results


# ── 8.3: Export labeled samples as CSV ────────────────────────────────────────

@router.get("/training-data/export")
async def export_training_data():
    """Stream all labeled training samples as a CSV file."""
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "sample_id", "video_title", "predicted_category",
        "confidence", "human_label", "transcript_preview", "ocr_preview",
    ])
    writer.writeheader()

    labeled_count = 0
    for path in sorted(TRAINING_DIR.glob("*.json")):
        try:
            text = path.read_text(encoding="utf-8").strip()
            data = json.loads(text) if text else {}
        except (json.JSONDecodeError, OSError):
            continue

        if not data.get("human_label"):
            continue  # skip unlabeled samples

        writer.writerow({
            "sample_id": path.stem,
            "video_title": data.get("video_title", ""),
            "predicted_category": data.get("predicted_category", ""),
            "confidence": data.get("confidence", ""),
            "human_label": data.get("human_label", ""),
            "transcript_preview": (data.get("transcript_preview") or "")[:300],
            "ocr_preview": (data.get("ocr_preview") or "")[:300],
        })
        labeled_count += 1

    logger.info(f"[admin] Exported {labeled_count} labeled training samples as CSV")

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=training_data.csv"},
    )
