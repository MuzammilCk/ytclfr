"""
services/vision/yolo_detector.py

YOLO-based object detection service.
Detects products and objects in video frames using the ultralytics YOLOv8 model.

Model is loaded once per worker process (singleton). Heavy inference is
offloaded to a thread executor to avoid blocking the asyncio event loop.
"""
from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger

from core.config import get_settings

settings = get_settings()

# Per-box confidence threshold — can be overridden via env
CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE", str(settings.YOLO_CONFIDENCE)))

# One process-level executor for CPU-bound YOLO inference
_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo")

# Module-level model singleton — loaded once on first use
_model: Optional[object] = None


def _load_model() -> object:
    """Load the YOLOv8n model (lazy, cached per process)."""
    global _model
    if _model is None:
        try:
            from ultralytics import YOLO  # noqa: PLC0415
            model_path = settings.YOLO_MODEL_PATH
            logger.info(f"Loading YOLO model from {model_path}")
            _model = YOLO(model_path)
            logger.info("YOLO model loaded successfully")
        except ImportError:
            logger.warning(
                "ultralytics package not installed — YOLO detection unavailable. "
                "Run: pip install ultralytics"
            )
            _model = None
        except Exception as exc:
            logger.error(f"Failed to load YOLO model: {exc}")
            _model = None
    return _model


@dataclass
class Detection:
    """A single object detected by YOLO in a video frame."""
    label: str                      # COCO class name, e.g. "bottle", "laptop"
    confidence: float               # 0.0 – 1.0
    frame_path: str                 # absolute path to the source frame
    bbox: List[float] = field(default_factory=list)  # [x1, y1, x2, y2] pixels


def _run_detection(frame_paths: List[str], conf_threshold: float) -> List[Detection]:
    """
    Run YOLOv8 inference synchronously.
    Prefers the model pre-loaded by worker_process_init; falls back to lazy load.
    """
    # Try to get the model from the pipeline's shared cache first
    try:
        from services.pipeline import _models  # noqa: PLC0415
        model = _models.get("yolo")
    except Exception:
        model = None

    if model is None:
        model = _load_model()
    if model is None:
        return []

    detections: List[Detection] = []
    for frame_path in frame_paths:
        try:
            results = model.predict(
                source=frame_path,
                conf=conf_threshold,
                verbose=False,
                save=False,
            )
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    conf = float(box.conf.item() if hasattr(box.conf, 'item') else box.conf[0])
                    if conf < conf_threshold:
                        continue
                    label_idx = int(box.cls[0])
                    label = model.names.get(label_idx, str(label_idx))
                    xyxy = box.xyxy[0]
                    bbox = xyxy.tolist() if hasattr(xyxy, 'tolist') else list(xyxy)
                    detections.append(Detection(
                        label=label,
                        confidence=round(conf, 3),
                        frame_path=frame_path,
                        bbox=bbox,
                    ))
        except Exception as exc:
            logger.warning(f"YOLO detection failed for frame {frame_path}: {exc}")

    return detections


class YOLODetector:
    """
    Async-friendly YOLO object detector.

    Usage::

        detector = YOLODetector()
        detections = await detector.detect(frame_paths)
    """

    def __init__(self, conf_threshold: float = CONFIDENCE_THRESHOLD) -> None:
        self._conf = conf_threshold

    async def detect(self, frame_paths: List[str]) -> List[Detection]:
        """
        Detect objects in a list of frame image files.

        Inference runs in a thread pool so the asyncio event loop stays
        non-blocking. Returns an empty list if the model cannot be loaded.

        Args:
            frame_paths: Absolute paths to frame image files (JPEG/PNG).

        Returns:
            Flat list of Detection objects from all supplied frames.
        """
        if not frame_paths:
            return []

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor,
            _run_detection,
            frame_paths,
            self._conf,
        )

    def is_available(self) -> bool:
        """Return True if the YOLO model loaded successfully."""
        return _load_model() is not None
