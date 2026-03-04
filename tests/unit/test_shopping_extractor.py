"""
tests/unit/test_shopping_extractor.py

Unit tests for the ShoppingExtractor.
No GPU, no network — all external services are mocked.
Run with: pytest tests/unit/test_shopping_extractor.py -v --tb=short
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from services.extraction.extractors import ShoppingExtractor
from services.vision.yolo_detector import Detection


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_extractor(detections=None):
    extractor = ShoppingExtractor()
    extractor.detections = detections or []
    return extractor


def _seg(text: str, start: float = 0.0) -> dict:
    return {"text": text, "start": start, "end": start + 2.0, "no_speech_prob": 0.05}


def _detection(label: str, conf: float = 0.85, frame: str = "/tmp/frame_001.jpg"):
    return Detection(label=label, confidence=conf, frame_path=frame, bbox=[0, 0, 100, 100])


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestShoppingExtractor:
    """Tests for ShoppingExtractor.extract() and helpers."""

    def test_output_type_is_shopping(self):
        """Result dict must have type == 'shopping'."""
        extractor = _make_extractor()
        with patch("services.extraction.extractors._get_nlp") as mock_nlp:
            mock_nlp.return_value.return_value = MagicMock(ents=[], noun_chunks=[])
            result = extractor.extract("buy this laptop today", [], {}, [], [])
        assert result["type"] == "shopping"

    def test_yolo_product_detected(self):
        """A laptop detection should appear in the products list."""
        det = _detection("laptop", conf=0.92)
        extractor = _make_extractor([det])

        with patch("services.extraction.extractors._get_nlp") as mock_nlp:
            mock_nlp.return_value.return_value = MagicMock(ents=[], noun_chunks=[])
            result = extractor.extract("", [], {}, ["/tmp/frame_001.jpg"], [])

        products = result["products"]
        assert len(products) == 1
        assert products[0]["name"] == "Laptop"
        assert products[0]["detection_source"] == "yolo"
        assert products[0]["confidence"] == 0.92

    def test_non_product_coco_label_excluded(self):
        """Person, car, etc. should NOT appear in the products list."""
        person_det = _detection("person", conf=0.99)
        extractor = _make_extractor([person_det])

        with patch("services.extraction.extractors._get_nlp") as mock_nlp:
            mock_nlp.return_value.return_value = MagicMock(ents=[], noun_chunks=[])
            result = extractor.extract("", [], {}, [], [])

        assert result["products"] == []

    def test_duplicate_frames_deduplicated(self):
        """Same label seen in 3 frames → 1 product with 3 frame_timestamps."""
        dets = [
            _detection("cell phone", conf=0.80, frame="/tmp/f1.jpg"),
            _detection("cell phone", conf=0.90, frame="/tmp/f2.jpg"),
            _detection("cell phone", conf=0.75, frame="/tmp/f3.jpg"),
        ]
        extractor = _make_extractor(dets)

        with patch("services.extraction.extractors._get_nlp") as mock_nlp:
            mock_nlp.return_value.return_value = MagicMock(ents=[], noun_chunks=[])
            result = extractor.extract("", [], {}, [], [])

        products = result["products"]
        assert len(products) == 1
        assert len(products[0]["frame_timestamps"]) == 3
        # Should keep the highest confidence
        assert products[0]["confidence"] == 0.90

    def test_google_shopping_url_format(self):
        """search_url must be a Google Shopping URL with the encoded label."""
        det = _detection("laptop")
        extractor = _make_extractor([det])

        with patch("services.extraction.extractors._get_nlp") as mock_nlp:
            mock_nlp.return_value.return_value = MagicMock(ents=[], noun_chunks=[])
            result = extractor.extract("", [], {}, [], [])

        url = result["products"][0]["search_url"]
        assert url.startswith("https://www.google.com/search?tbm=shop")
        assert "laptop" in url

    def test_total_products_matches_list(self):
        """total_products field must equal len(products)."""
        dets = [_detection("laptop"), _detection("cell phone")]
        extractor = _make_extractor(dets)

        with patch("services.extraction.extractors._get_nlp") as mock_nlp:
            mock_nlp.return_value.return_value = MagicMock(ents=[], noun_chunks=[])
            result = extractor.extract("", [], {}, [], [])

        assert result["total_products"] == len(result["products"])

    def test_category_inference_electronics(self):
        """Known electronics labels should map to 'Electronics' category."""
        det = _detection("tv", conf=0.88)
        extractor = _make_extractor([det])

        with patch("services.extraction.extractors._get_nlp") as mock_nlp:
            mock_nlp.return_value.return_value = MagicMock(ents=[], noun_chunks=[])
            result = extractor.extract("", [], {}, [], [])

        assert result["products"][0]["category"] == "Electronics"

    def test_no_detections_no_products(self):
        """With zero detections and no NLP entities, products list is empty."""
        extractor = _make_extractor([])

        with patch("services.extraction.extractors._get_nlp") as mock_nlp:
            mock_nlp.return_value.return_value = MagicMock(ents=[], noun_chunks=[])
            result = extractor.extract("", [], {}, [], [])

        assert result["products"] == []
        assert result["total_products"] == 0
