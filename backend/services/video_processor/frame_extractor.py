"""
services/video_processor/frame_extractor.py

Extracts key frames from a downloaded video using OpenCV.
Implements adaptive sampling: lower FPS for long videos to keep
memory and compute cost under control, while always extracting
enough frames for reliable classification.
"""
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from core.config import get_settings

settings = get_settings()


@dataclass
class FrameExtractionResult:
    frame_paths: List[str]
    total_frames_extracted: int
    video_fps: float
    duration_secs: float
    resolution: Tuple[int, int]   # (width, height)
    sample_interval_secs: float


class FrameExtractor:
    """
    OpenCV-based frame sampler.

    Adaptive sampling strategy
    ─────────────────────────
    Duration        → sample rate
    ≤ 5 min         → 2 FPS
    5 – 20 min      → 1 FPS  (default)
    20 – 60 min     → 0.5 FPS
    > 60 min        → 0.2 FPS
    Always extracts ≥ 30 and ≤ 1800 frames per video.
    """

    MIN_FRAMES = 30
    MAX_FRAMES = 1800

    def __init__(self, frames_dir: Optional[str] = None):
        self.frames_dir = Path(frames_dir or settings.FRAMES_DIR)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    async def extract(self, video_path: str, video_id: str) -> FrameExtractionResult:
        """Async wrapper — OpenCV is CPU-bound so we offload to thread pool."""
        return await asyncio.get_running_loop().run_in_executor(
            None, self._extract_sync, video_path, video_id
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _extract_sync(self, video_path: str, video_id: str) -> FrameExtractionResult:
        out_dir = self.frames_dir / video_id
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {video_path}")

        video_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frame_count: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_secs = total_frame_count / video_fps

        target_fps = self._adaptive_fps(duration_secs)
        frame_interval = max(1, int(video_fps / target_fps))

        logger.info(
            f"[{video_id}] {duration_secs:.0f}s video @ {video_fps:.1f}fps → "
            f"sampling every {frame_interval} frames ({target_fps:.2f} FPS)"
        )

        frame_paths: List[str] = []
        frame_count = 0
        saved_count = 0
        prev_gray: Optional[np.ndarray] = None

        while cap.isOpened() and saved_count < self.MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                # Scene-change filter: skip visually near-identical frames
                if self._is_significant(frame, prev_gray):
                    path = out_dir / f"frame_{saved_count:05d}.jpg"
                    cv2.imwrite(
                        str(path),
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 85],
                    )
                    frame_paths.append(str(path))
                    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    saved_count += 1

            frame_count += 1

        cap.release()

        # Safety net: if fewer than MIN_FRAMES saved, re-extract without filter
        if len(frame_paths) < self.MIN_FRAMES:
            logger.warning(
                f"[{video_id}] Only {len(frame_paths)} unique frames; "
                "re-extracting without scene-change filter"
            )
            frame_paths = self._force_extract(video_path, video_id, out_dir, video_fps)

        logger.info(f"[{video_id}] Extracted {len(frame_paths)} frames")
        return FrameExtractionResult(
            frame_paths=frame_paths,
            total_frames_extracted=len(frame_paths),
            video_fps=video_fps,
            duration_secs=duration_secs,
            resolution=(width, height),
            sample_interval_secs=frame_interval / video_fps,
        )

    @staticmethod
    def _adaptive_fps(duration_secs: float) -> float:
        if duration_secs <= 300:
            return 2.0
        if duration_secs <= 1200:
            return 1.0
        if duration_secs <= 3600:
            return 0.5
        return 0.2

    @staticmethod
    def _is_significant(frame: np.ndarray, prev_gray: Optional[np.ndarray]) -> bool:
        """Return True if frame differs enough from the previous saved frame."""
        if prev_gray is None:
            return True
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        score = float(diff.mean())
        return score > 3.0   # empirically tuned threshold

    def _force_extract(
        self,
        video_path: str,
        video_id: str,
        out_dir: Path,
        video_fps: float,
    ) -> List[str]:
        """Re-extract uniformly spaced frames ignoring scene-change filter."""
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_frames = min(self.MIN_FRAMES, total)
        step = max(1, total // n_frames)

        paths: List[str] = []
        idx = 0
        while cap.isOpened() and len(paths) < n_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            path = out_dir / f"force_{len(paths):05d}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            paths.append(str(path))
            idx += step

        cap.release()
        return paths

    def cleanup(self, video_id: str):
        """Delete all extracted frames for a given video ID."""
        frame_dir = self.frames_dir / video_id
        if frame_dir.exists():
            for f in frame_dir.glob("*.jpg"):
                f.unlink(missing_ok=True)
            frame_dir.rmdir()
