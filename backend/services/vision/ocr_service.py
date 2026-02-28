"""
services/vision/ocr_service.py

Tesseract-based OCR for extracting text overlaid on video frames.
Useful for:
  - Listicle videos that show numbered lists on screen
  - Tutorial videos showing code or step-by-step instructions
  - Music videos that show track names / artist credits
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from loguru import logger

try:
    import pytesseract
    from PIL import Image
    _TESSERACT_AVAILABLE = True
    # Set Tesseract executable path from env/settings (Windows needs this)
    from core.config import get_settings as _get_settings
    _settings = _get_settings()
    _cmd = _settings.TESSERACT_CMD
    if _cmd:
        pytesseract.pytesseract.tesseract_cmd = _cmd
except ImportError:
    _TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not installed; OCR will be disabled")


@dataclass
class OCRResult:
    frame_path: str
    timestamp_secs: Optional[float]
    raw_text: str
    cleaned_text: str
    confidence: float   # 0-100, mean Tesseract word confidence


class OCRService:
    """
    Extracts text from video frames using Tesseract OCR.
    Applies preprocessing (grayscale → CLAHE → threshold → denoise)
    to maximise OCR accuracy on diverse backgrounds.
    """

    # Regex to filter out single chars, random symbols, and very short junk
    _NOISE_RE = re.compile(r"[^\w\s\-.,!?:/()'\"@#&*+=]")
    _MIN_WORD_LEN = 2
    # Minimum Laplacian variance — frames below this are considered too blurry for OCR
    _BLUR_THRESHOLD = float(os.getenv("OCR_TEXT_THRESHOLD", "100.0"))

    def __init__(self, lang: Optional[str] = None):
        _s = get_settings() if 'get_settings' in dir() else None
        self.lang = lang or (os.getenv("OCR_LANG") or (_s.OCR_LANG if _s else "eng"))
        self.tesseract_config = "--oem 3 --psm 6"   # OEM3=LSTM, PSM6=assume uniform block of text

    # ── Public API ─────────────────────────────────────────────────────────────

    async def extract_from_frames(
        self,
        frame_paths: List[str],
        max_frames: int = 20,
        fps: float = 1.0,
    ) -> List[OCRResult]:
        """
        Run OCR on a sample of frames. Returns one result per frame.
        Offloads CPU-heavy OCR to a thread pool.
        """
        if not _TESSERACT_AVAILABLE:
            return []

        # Sample evenly across the frame list
        step = max(1, len(frame_paths) // max_frames)
        sampled = frame_paths[::step][:max_frames]

        results = await asyncio.gather(
            *[
                asyncio.get_event_loop().run_in_executor(
                    None, self._ocr_frame, path, i / fps
                )
                for i, path in enumerate(sampled)
            ],
            return_exceptions=True,
        )

        return [r for r in results if isinstance(r, OCRResult) and r.cleaned_text.strip()]

    def aggregate_text(self, results: List[OCRResult]) -> str:
        """Merge OCR results from multiple frames into a single deduplicated string."""
        seen: set = set()
        lines: List[str] = []
        for r in results:
            for line in r.cleaned_text.split("\n"):
                line = line.strip()
                if line and line.lower() not in seen and len(line) > self._MIN_WORD_LEN:
                    seen.add(line.lower())
                    lines.append(line)
        return "\n".join(lines)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _ocr_frame(self, frame_path: str, timestamp_secs: float) -> Optional[OCRResult]:
        """Synchronous Tesseract call — runs in thread pool."""
        try:
            img = cv2.imread(str(frame_path))
            if img is None:
                return None

            # Blurriness check — skip frames with insufficient texture for OCR
            gray_check = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray_check, cv2.CV_64F).var()
            if laplacian_var < self._BLUR_THRESHOLD:
                return None

            preprocessed = self._preprocess(img)
            pil_img = Image.fromarray(preprocessed)

            # Get data with per-word confidence
            data = pytesseract.image_to_data(
                pil_img,
                lang=self.lang,
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT,
            )

            words = []
            confidences = []
            for i, word in enumerate(data["text"]):
                conf = int(data["conf"][i])
                if conf > 30 and word.strip():   # filter low-confidence and empty tokens
                    words.append(word)
                    confidences.append(conf)

            raw_text = " ".join(words)
            cleaned  = self._clean(raw_text)
            avg_conf = float(np.mean(confidences)) if confidences else 0.0

            return OCRResult(
                frame_path=str(frame_path),
                timestamp_secs=timestamp_secs,
                raw_text=raw_text,
                cleaned_text=cleaned,
                confidence=avg_conf,
            )
        except Exception as exc:
            logger.warning(f"OCR failed on {frame_path}: {exc}")
            return None

    @staticmethod
    def _preprocess(img: np.ndarray) -> np.ndarray:
        """
        Image preprocessing pipeline optimised for text detection:
        1. Convert to grayscale
        2. CLAHE (contrast-limited adaptive histogram equalisation)
        3. Gaussian denoising
        4. Otsu thresholding → binary
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Upscale small frames for better OCR
        h, w = gray.shape
        if w < 640:
            scale = 640 / w
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)

        # Denoise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _clean(self, text: str) -> str:
        """Remove noise characters and filter very short tokens."""
        text = self._NOISE_RE.sub(" ", text)
        words = [w for w in text.split() if len(w) >= self._MIN_WORD_LEN]
        return " ".join(words)
