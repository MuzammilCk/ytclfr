"""
services/classification/classifier.py

Multi-modal ensemble classifier.

Modalities
──────────
1. Vision  – EfficientNet-B0 fine-tuned on per-frame features
2. Text    – BERT (bert-base-uncased) on transcript + metadata
3. Heuristic signals – title keywords, description patterns, tag analysis

Final prediction is a weighted ensemble of the three sub-classifiers.

Categories (8 classes)
───────────────────────
comedy | listicle | music | educational | news | review | gaming | vlog
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from core.config import get_settings as _cfg
MAX_FRAMES = int(os.getenv("MAX_FRAMES", str(_cfg().MAX_FRAMES)))
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from PIL import Image
from transformers import BertModel, BertTokenizer
from torchvision import models, transforms

from core.config import get_settings

settings = get_settings()

# ── Label definitions ─────────────────────────────────────────────────────────
CATEGORIES = ["comedy", "listicle", "music", "educational", "news", "review", "gaming", "vlog", "shopping"]
N_CLASSES = len(CATEGORIES)


@dataclass
class ClassificationOutput:
    predicted_category: str
    confidence: float
    all_scores: Dict[str, float]
    modality_breakdown: Dict[str, float] = field(default_factory=dict)


# ── Vision sub-model ──────────────────────────────────────────────────────────

class FrameClassifier(nn.Module):
    """EfficientNet-B0 with replaced head for 8-class video category classification."""

    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, n_classes),
        )
        self.backbone = backbone

        # Load local fine-tuned weights if available
        ckpt_path = Path("checkpoints/best_frame_model.pth")
        if ckpt_path.exists():
            try:
                state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                # Unwrap from nn.DataParallel/DDP if necessary
                if list(state_dict.keys())[0].startswith("module."):
                    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                self.load_state_dict(state_dict)
                logger.info("ytclfr-classify: Loaded fine-tuned vision weights.")
            except Exception as e:
                logger.warning(f"ytclfr-classify: Failed to load vision checkpoint: {e}")
        else:
            logger.warning("ytclfr-classify: Vision checkpoint not found. Using untuned backbone.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


_FRAME_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Text sub-model ────────────────────────────────────────────────────────────

class TextClassifier(nn.Module):
    """BERT encoder + classification head."""

    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        hidden = self.bert.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

        # Load local fine-tuned weights if available
        ckpt_path = Path("checkpoints/best_text_model.pth")
        if ckpt_path.exists():
            try:
                state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                if list(state_dict.keys())[0].startswith("module."):
                    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                self.load_state_dict(state_dict)
                logger.info("ytclfr-classify: Loaded fine-tuned text weights.")
            except Exception as e:
                logger.warning(f"ytclfr-classify: Failed to load text checkpoint: {e}")
        else:
            logger.warning("ytclfr-classify: Text checkpoint not found. Using untuned backbone.")

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = out.pooler_output          # [CLS] representation
        return self.classifier(pooled)


# ── Heuristic rules ───────────────────────────────────────────────────────────

_HEURISTIC_PATTERNS: Dict[str, List[re.Pattern]] = {
    "listicle": [
        re.compile(r"\btop\s*\d+\b", re.I),
        re.compile(r"\bbest\s+\d+\b", re.I),
        re.compile(r"\b\d+\s+best\b", re.I),
        re.compile(r"\branking\b", re.I),
        re.compile(r"\bevery\s+\w+\s+ranked\b", re.I),
        # Rating-style: "10/10 movies", "5/5 games"
        re.compile(r"\b\d+/\d+\s+(movies?|films?|shows?|games?|songs?|books?)\b", re.I),
        # "best X of all time", "greatest X ever made"  
        re.compile(r"\bbest\s+.{1,30}(of all time|ever|ever made)\b", re.I),
        re.compile(r"\bgreatest\s+.{1,30}(of all time|ever)\b", re.I),
        # "must watch", "must see"
        re.compile(r"\bmust[\s-]?(watch|see|read)\b", re.I),
        # "only perfect movies", "iconic films", "perfect films"
        re.compile(r"\b(only|perfect|greatest|iconic)\s+.{1,20}(movies?|films?)\b", re.I),
        # Movies/films with action word
        re.compile(r"\b(movies?|films?)\s+(you|to|list|ranked|worth|that)\b", re.I),
    ],
    "music": [
        re.compile(r"\bplaylist\b", re.I),
        re.compile(r"\bmix\b", re.I),
        re.compile(r"\bsongs\b", re.I),
        re.compile(r"\bhits\b", re.I),
        re.compile(r"\balbum\b", re.I),
    ],
    "educational": [
        re.compile(r"\btutorial\b", re.I),
        re.compile(r"\bhow[\s-]?to\b", re.I),
        re.compile(r"\blearn\b", re.I),
        re.compile(r"\bcourse\b", re.I),
        re.compile(r"\bexplain\b", re.I),
        re.compile(r"\blecture\b", re.I),
    ],
    "comedy": [
        re.compile(r"\bskit\b", re.I),
        re.compile(r"\bprank\b", re.I),
        re.compile(r"\bstand[\s-]?up\b", re.I),
        re.compile(r"\bcomedy\b", re.I),
        re.compile(r"\bfunny\b", re.I),
    ],
    "gaming": [
        re.compile(r"\bgameplay\b", re.I),
        re.compile(r"\blet['']s play\b", re.I),
        re.compile(r"\bwalkthrough\b", re.I),
        re.compile(r"\bspeedrun\b", re.I),
        re.compile(r"\besports\b", re.I),
    ],
    "review": [
        re.compile(r"\breview\b", re.I),
        re.compile(r"\bunboxing\b", re.I),
        re.compile(r"\bbenchmark\b", re.I),
        re.compile(r"\bvs\b", re.I),
    ],
    "news": [
        re.compile(r"\bbreaking\b", re.I),
        re.compile(r"\bnews\b", re.I),
        re.compile(r"\breport\b", re.I),
        re.compile(r"\banalysis\b", re.I),
    ],
    "vlog": [
        re.compile(r"\bvlog\b", re.I),
        re.compile(r"\bday in (my|the) life\b", re.I),
        re.compile(r"\bwith me\b", re.I),
    ],
    "shopping": [
        re.compile(r"\bshop\b", re.I),
        re.compile(r"\bshopping\b", re.I),
        re.compile(r"\bproduct\b", re.I),
        re.compile(r"\bbuy\b", re.I),
        re.compile(r"\bprice\b", re.I),
        re.compile(r"\bhaul\b", re.I),
        re.compile(r"\baffiliate\b", re.I),
        re.compile(r"\blink in bio\b", re.I),
    ],
}

ENSEMBLE_WEIGHTS = {
    "vision": 0.40,
    "text": 0.40,
    "heuristic": 0.20,
}


# ── Main classifier ───────────────────────────────────────────────────────────

class MultiModalClassifier:
    """
    Loads (or initialises) frame + text sub-models and combines their
    predictions into a final ensemble score.

    In production, load fine-tuned weights from disk.
    If no weights found, models start from their ImageNet/BERT pretrained
    state (provides reasonable zero-shot proxy via transfer learning).
    """

    def __init__(self):
        self.device = torch.device(settings.TORCH_DEVICE)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # Models are NOT instantiated here — they are accessed from the
        # worker_process_init cache in services.pipeline._models.
        # Fallback to a locally loaded instance only if the cache is absent.
        self._frame_model: Optional[FrameClassifier] = None
        self._text_model: Optional[TextClassifier] = None

    def _get_frame_model(self) -> FrameClassifier:
        """Return the cached frame model or create a fallback instance."""
        try:
            from services.pipeline import _models  # noqa: PLC0415
            if "efficientnet" in _models:
                return _models["efficientnet"]
        except Exception:
            pass
        if self._frame_model is None:
            self._frame_model = FrameClassifier().to(self.device)
            self._load_frame_weights(self._frame_model)
            self._frame_model.eval()
        return self._frame_model

    def _get_text_model(self) -> TextClassifier:
        """Return the cached text model or create a fallback instance."""
        try:
            from services.pipeline import _models  # noqa: PLC0415
            if "bert" in _models:
                return _models["bert"]
        except Exception:
            pass
        if self._text_model is None:
            self._text_model = TextClassifier().to(self.device)
            self._load_text_weights(self._text_model)
            self._text_model.eval()
        return self._text_model

    def _load_frame_weights(self, model: FrameClassifier):
        ckpt = Path(settings.DOWNLOAD_DIR).parent / "models" / "frame_classifier.pt"
        if ckpt.exists():
            model.load_state_dict(torch.load(str(ckpt), map_location=self.device))
            logger.info("Loaded fine-tuned frame classifier weights")
        else:
            logger.warning("No fine-tuned frame weights found; using ImageNet pretrained")

    def _load_text_weights(self, model: TextClassifier):
        ckpt = Path(settings.DOWNLOAD_DIR).parent / "models" / "text_classifier.pt"
        if ckpt.exists():
            model.load_state_dict(torch.load(str(ckpt), map_location=self.device))
            logger.info("Loaded fine-tuned text classifier weights")
        else:
            logger.warning("No fine-tuned text weights found; using BERT pretrained")

    # ── Sub-classifiers ───────────────────────────────────────────────────────

    @torch.no_grad()
    def classify_frames(self, frame_paths: List[str]) -> np.ndarray:
        """
        Score each frame and return average softmax distribution.
        Returns uniform prior if frame_paths is empty.
        """
        sampled = frame_paths[:MAX_FRAMES] if frame_paths else []
        if not sampled:
            return np.ones(N_CLASSES) / N_CLASSES   # uniform prior — text-only fallback

        frame_model = self._get_frame_model()
        tensors: List[torch.Tensor] = []
        for p in sampled:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(_FRAME_TRANSFORM(img))
            except Exception:
                continue

        if not tensors:
            return np.ones(N_CLASSES) / N_CLASSES

        batch = torch.stack(tensors).to(self.device)      # (N, 3, 224, 224)
        logits = frame_model(batch)                        # (N, 8)
        probs = torch.softmax(logits, dim=1)               # (N, 8)
        return probs.mean(dim=0).cpu().numpy()             # (8,)

    @torch.no_grad()
    def classify_text(self, text: str) -> np.ndarray:
        """Encode title + transcript with BERT and return softmax distribution.
        Returns uniform prior if transcript is empty."""
        if not text.strip():
            return np.ones(N_CLASSES) / N_CLASSES   # uniform prior — vision-only fallback

        text_model = self._get_text_model()
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        logits = text_model(**encoded)            # (1, 8)
        probs = torch.softmax(logits, dim=1)
        return probs.squeeze(0).cpu().numpy()     # (8,)

    @staticmethod
    def classify_heuristic(title: str, description: str, tags: List[str]) -> np.ndarray:
        """
        Pattern-matching heuristic: count regex hits per category,
        convert to a probability distribution.
        """
        combined = " ".join([title, description, *tags])
        scores = np.zeros(N_CLASSES)

        for cat, patterns in _HEURISTIC_PATTERNS.items():
            cat_idx = CATEGORIES.index(cat)
            for pat in patterns:
                if pat.search(combined):
                    scores[cat_idx] += 1.0

        total = scores.sum()
        if total == 0:
            return np.ones(N_CLASSES) / N_CLASSES
        return scores / total

    # ── Ensemble ──────────────────────────────────────────────────────────────

    def predict(
        self,
        frame_paths: List[str],
        transcript: str,
        title: str,
        description: str,
        tags: List[str],
        ocr_text: str = "",
    ) -> ClassificationOutput:
        """
        Run all three sub-classifiers and return a weighted ensemble prediction.
        Falls back to text-only if frame_paths is empty, or vision-only if transcript is empty.
        Returns 'unknown' if max confidence < 0.25.
        """
        text_input = f"{title}. {ocr_text}. {transcript[:1000]}"  # title always first

        # Guarded calls — each returns uniform prior when input is missing
        vision_probs = self.classify_frames(frame_paths)   # uniform if no frames
        text_probs = self.classify_text(text_input)         # uniform if no transcript
        heuristic_probs = self.classify_heuristic(title, description, tags)
        
        # If transcript is very thin, run heuristic over OCR as well and take element-wise max
        if len(transcript.split()) < 20 and ocr_text:
            ocr_heuristic_probs = self.classify_heuristic(title, description, [ocr_text])
            heuristic_probs = np.maximum(heuristic_probs, ocr_heuristic_probs)
            # Re-normalize
            total = heuristic_probs.sum()
            if total > 0:
                heuristic_probs /= total

        # Adjust weights dynamically to avoid diluting the available signal
        w_vision = ENSEMBLE_WEIGHTS["vision"] if frame_paths else 0.0
        w_text = ENSEMBLE_WEIGHTS["text"] if text_input.strip() else 0.0
        w_heuristic = ENSEMBLE_WEIGHTS["heuristic"]
        total_w = w_vision + w_text + w_heuristic or 1.0

        combined = (
            (w_vision / total_w) * vision_probs
            + (w_text / total_w) * text_probs
            + (w_heuristic / total_w) * heuristic_probs
        )

        pred_idx = int(combined.argmax())
        confidence = float(combined[pred_idx])

        all_scores = {CATEGORIES[i]: float(combined[i]) for i in range(N_CLASSES)}

        # Low-confidence fallback (0.25 to prevent shorts dropping to unknown)
        if confidence < 0.25:
            return ClassificationOutput(
                predicted_category="unknown",
                confidence=confidence,
                all_scores=all_scores,
                modality_breakdown={},
            )

        return ClassificationOutput(
            predicted_category=CATEGORIES[pred_idx],
            confidence=confidence,
            all_scores=all_scores,
            modality_breakdown={
                "vision": float(vision_probs[pred_idx]),
                "text": float(text_probs[pred_idx]),
                "heuristic": float(heuristic_probs[pred_idx]),
            },
        )
