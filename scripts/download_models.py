"""
scripts/download_models.py

Downloads all ML model checkpoints required by the YouTube Intelligent Classifier.

Models downloaded:
  - Whisper (base) — OpenAI speech-to-text
  - YOLOv8n         — Ultralytics object detection

Usage:
    cd scripts
    python download_models.py            # downloads all
    python download_models.py --whisper  # Whisper only
    python download_models.py --yolo     # YOLO only

All downloads go to their respective library cache directories.
No extra paths are created — both libraries manage their own caches.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _hr():
    print("-" * 60)


def _ok(msg: str):
    print(f"  \u2714  {msg}")


def _fail(msg: str):
    print(f"  \u2718  {msg}", file=sys.stderr)


# ── Whisper ───────────────────────────────────────────────────────────────────

def download_whisper(model_size: str = "base") -> bool:
    """
    Download the specified Whisper model using openai-whisper's load_model().
    The model is cached in ~/.cache/whisper on Linux/macOS or
    %USERPROFILE%\.cache\whisper on Windows.

    Args:
        model_size: One of tiny | base | small | medium | large

    Returns:
        True on success, False on failure.
    """
    _hr()
    print(f"Downloading Whisper model: {model_size}")
    try:
        import whisper  # noqa: PLC0415
        t0 = time.perf_counter()
        model = whisper.load_model(model_size)
        elapsed = time.perf_counter() - t0
        # Estimate file size via parameter count
        param_count = sum(p.numel() for p in model.parameters())
        size_mb = param_count * 4 / 1024 / 1024   # float32 approx
        _ok(f"Whisper '{model_size}' ready  (~{size_mb:.0f} MB parameters, {elapsed:.1f}s)")
        return True
    except ImportError:
        _fail("openai-whisper not installed. Run: pip install openai-whisper")
        return False
    except Exception as exc:
        _fail(f"Whisper download failed: {exc}")
        return False


# ── YOLOv8 ────────────────────────────────────────────────────────────────────

def download_yolo(model_name: str = "yolov8n.pt") -> bool:
    """
    Download the YOLOv8n model weights using ultralytics.
    The model file is saved to the current directory if not already present,
    or to the ultralytics cache (~/.cache/ultralytics/).

    Args:
        model_name: Model filename, e.g. yolov8n.pt, yolov8s.pt

    Returns:
        True on success, False on failure.
    """
    _hr()
    print(f"Downloading YOLO model: {model_name}")
    try:
        from ultralytics import YOLO  # noqa: PLC0415
        t0 = time.perf_counter()
        model = YOLO(model_name)  # downloads on first call if not cached
        elapsed = time.perf_counter() - t0

        # Find the cached file to report its size
        model_path = Path(model_name)
        if not model_path.exists():
            # Try ultralytics default cache
            cache = Path.home() / ".cache" / "ultralytics" / model_name
            model_path = cache if cache.exists() else Path(model_name)

        if model_path.exists():
            size_mb = model_path.stat().st_size / 1024 / 1024
            _ok(f"YOLO '{model_name}' ready  ({size_mb:.1f} MB, {elapsed:.1f}s) → {model_path}")
        else:
            _ok(f"YOLO '{model_name}' ready  ({elapsed:.1f}s)")

        # Quick smoke test
        _ = model.names   # verifies the model object is usable
        _ok("YOLO smoke test passed (model.names accessible)")
        return True
    except ImportError:
        _fail("ultralytics not installed. Run: pip install ultralytics")
        return False
    except Exception as exc:
        _fail(f"YOLO download failed: {exc}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download ML model weights for YouTube Classifier"
    )
    parser.add_argument("--whisper", action="store_true", help="Download Whisper only")
    parser.add_argument("--yolo",    action="store_true", help="Download YOLO only")
    parser.add_argument(
        "--whisper-size", default=os.getenv("WHISPER_MODEL_SIZE", "base"),
        metavar="SIZE", help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--yolo-model", default=os.getenv("YOLO_MODEL_PATH", "yolov8n.pt"),
        metavar="NAME", help="YOLO model filename (default: yolov8n.pt)"
    )
    args = parser.parse_args()

    # If no specific flag, download everything
    download_all = not (args.whisper or args.yolo)

    print("=" * 60)
    print("YouTube Classifier — Model Downloader")
    print("=" * 60)

    results: list[bool] = []

    if download_all or args.whisper:
        results.append(download_whisper(args.whisper_size))

    if download_all or args.yolo:
        results.append(download_yolo(args.yolo_model))

    _hr()
    if all(results):
        print(f"\n\u2714 All {len(results)} model(s) ready. You can now start the backend.\n")
        return 0
    else:
        failed = results.count(False)
        print(f"\n\u2718 {failed}/{len(results)} model(s) failed. Check errors above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
