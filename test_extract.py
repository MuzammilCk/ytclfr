"""
test_extract.py
---------------
Standalone test for frame-by-frame extraction + YOLO detection.

Usage:
    python test_extract.py <path_to_video.mp4> [--conf 0.25] [--max-frames 60] [--out ./test_frames]

All dependencies come from the project's requirements.txt (opencv-python, ultralytics, numpy).
No app settings / Django / FastAPI imports are needed here.
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Config defaults (can be overridden via CLI)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MAX_FRAMES = 60
DEFAULT_MIN_FRAMES = 30
DEFAULT_CONF = 0.25
DEFAULT_OUT_DIR = "./test_frames"
DEFAULT_YOLO_MODEL = "yolov8n.pt"   # will auto-download if not present


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FrameExtractionResult:
    frame_paths: List[str]
    total_frames_extracted: int
    video_fps: float
    duration_secs: float
    resolution: Tuple[int, int]
    sample_interval_secs: float


@dataclass
class Detection:
    label: str
    confidence: float
    frame_path: str
    bbox: List[float] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Frame Extractor  (copied from services/video_processor/frame_extractor.py)
# ─────────────────────────────────────────────────────────────────────────────
def _adaptive_fps(duration_secs: float) -> float:
    if duration_secs <= 300:
        return 2.0
    if duration_secs <= 1200:
        return 1.0
    if duration_secs <= 3600:
        return 0.5
    return 0.2


def _is_significant(frame: np.ndarray, prev_gray: Optional[np.ndarray]) -> bool:
    """Return True if frame differs enough from the previous saved frame."""
    if prev_gray is None:
        return True
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, prev_gray)
    score = float(diff.mean())
    return score > 3.0   # empirically tuned threshold


def _force_extract(video_path: str, out_dir: Path, min_frames: int) -> List[str]:
    """Re-extract uniformly spaced frames ignoring scene-change filter."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(min_frames, total)
    step = max(1, total // n_frames) if n_frames > 0 else 1

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


def extract_frames(
    video_path: str,
    out_dir: Path,
    max_frames: int = DEFAULT_MAX_FRAMES,
    min_frames: int = DEFAULT_MIN_FRAMES,
) -> FrameExtractionResult:
    """Extract key frames from video_path into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    video_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frame_count: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_secs = total_frame_count / video_fps

    target_fps = _adaptive_fps(duration_secs)
    frame_interval = max(1, int(video_fps / target_fps))

    estimated_samples = total_frame_count // frame_interval
    if estimated_samples > max_frames:
        frame_interval = max(1, total_frame_count // max_frames)

    print(
        f"[extract] {duration_secs:.0f}s video @ {video_fps:.1f} fps → "
        f"sampling every {frame_interval} frames (cap={max_frames})"
    )

    frame_paths: List[str] = []
    frame_count = 0
    saved_count = 0
    prev_gray: Optional[np.ndarray] = None

    while cap.isOpened() and saved_count < max_frames and frame_count < total_frame_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            break

        if _is_significant(frame, prev_gray):
            path = out_dir / f"frame_{saved_count:05d}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_paths.append(str(path))
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            saved_count += 1

        frame_count += frame_interval

    cap.release()

    if len(frame_paths) < min_frames:
        print(
            f"[extract] Only {len(frame_paths)} unique frames found — "
            "re-extracting without scene-change filter..."
        )
        frame_paths = _force_extract(video_path, out_dir, min_frames)

    print(f"[extract] Done — extracted {len(frame_paths)} frames to: {out_dir}")
    return FrameExtractionResult(
        frame_paths=frame_paths,
        total_frames_extracted=len(frame_paths),
        video_fps=video_fps,
        duration_secs=duration_secs,
        resolution=(width, height),
        sample_interval_secs=frame_interval / video_fps,
    )


# ─────────────────────────────────────────────────────────────────────────────
# YOLO Detector  (copied from services/vision/yolo_detector.py)
# ─────────────────────────────────────────────────────────────────────────────
def run_yolo(
    frame_paths: List[str],
    conf_threshold: float = DEFAULT_CONF,
    model_path: str = DEFAULT_YOLO_MODEL,
) -> List[Detection]:
    """Run YOLOv8 detection on a list of frame image paths."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[yolo] ERROR: ultralytics is not installed. Run: pip install ultralytics")
        return []

    print(f"[yolo] Loading model: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as exc:
        print(f"[yolo] ERROR: Failed to load model — {exc}")
        return []

    detections: List[Detection] = []

    for i, frame_path in enumerate(frame_paths):
        try:
            results = model.predict(
                source=frame_path,
                conf=conf_threshold,
                verbose=False,
                save=False,
            )
            frame_dets = 0
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    conf = float(box.conf.item() if hasattr(box.conf, "item") else box.conf[0])
                    if conf < conf_threshold:
                        continue
                    label_idx = int(box.cls[0])
                    label = model.names.get(label_idx, str(label_idx))
                    xyxy = box.xyxy[0]
                    bbox = xyxy.tolist() if hasattr(xyxy, "tolist") else list(xyxy)
                    detections.append(
                        Detection(
                            label=label,
                            confidence=round(conf, 3),
                            frame_path=frame_path,
                            bbox=bbox,
                        )
                    )
                    frame_dets += 1

            print(f"  frame {i+1:>4}/{len(frame_paths)}  {os.path.basename(frame_path)}  → {frame_dets} detection(s)")
        except Exception as exc:
            print(f"  frame {i+1:>4}/{len(frame_paths)}  {os.path.basename(frame_path)}  → ERROR: {exc}")

    return detections


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Test frame extraction + YOLO detection on a local .mp4 file"
    )
    parser.add_argument("video", help="Path to the .mp4 (or any OpenCV-readable) video file")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help=f"YOLO confidence threshold (default: {DEFAULT_CONF})")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES, help=f"Max frames to extract (default: {DEFAULT_MAX_FRAMES})")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help=f"Output directory for frames (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--model", default=DEFAULT_YOLO_MODEL, help=f"YOLO model path or name (default: {DEFAULT_YOLO_MODEL})")
    parser.add_argument("--skip-yolo", action="store_true", help="Only extract frames, skip YOLO detection")
    args = parser.parse_args()

    video_path = args.video
    if not os.path.isfile(video_path):
        print(f"ERROR: File not found: {video_path}")
        sys.exit(1)

    out_dir = Path(args.out)

    # ── Step 1: Frame extraction ──────────────────────────────────────────────
    print("\n=== STEP 1: Frame Extraction ===")
    result = extract_frames(
        video_path=video_path,
        out_dir=out_dir,
        max_frames=args.max_frames,
    )
    print(f"\nSummary:")
    print(f"  Duration      : {result.duration_secs:.1f} s")
    print(f"  Video FPS     : {result.video_fps:.2f}")
    print(f"  Resolution    : {result.resolution[0]}x{result.resolution[1]}")
    print(f"  Sample interval: {result.sample_interval_secs:.2f} s")
    print(f"  Frames saved  : {result.total_frames_extracted}")
    print(f"  Output dir    : {out_dir.resolve()}")

    if args.skip_yolo:
        print("\n[--skip-yolo] Skipping YOLO detection.")
        return

    # ── Step 2: YOLO detection ─────────────────────────────────────────────────
    print("\n=== STEP 2: YOLO Detection ===")
    detections = run_yolo(
        frame_paths=result.frame_paths,
        conf_threshold=args.conf,
        model_path=args.model,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n=== YOLO Results ===")
    print(f"Total detections: {len(detections)}")

    if detections:
        from collections import Counter
        label_counts = Counter(d.label for d in detections)
        print("\nTop detected objects:")
        for label, count in label_counts.most_common(15):
            print(f"  {label:<20} {count:>4}x")

        print("\nTop-5 highest-confidence detections:")
        for d in sorted(detections, key=lambda x: x.confidence, reverse=True)[:5]:
            print(f"  [{d.confidence:.3f}] {d.label:<20} — {os.path.basename(d.frame_path)}")
    else:
        print("No objects detected (try lowering --conf threshold).")


if __name__ == "__main__":
    main()
