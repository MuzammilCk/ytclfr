"""
tests/unit/test_yolo_detector.py

Unit tests for the YOLO detector service.
Mocks the ultralytics.YOLO model so no GPU/model file is required.
Run with: pytest tests/unit/test_yolo_detector.py -v --tb=short
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_box(label_idx: int, conf: float, xyxy: list):
    """Build a fake ultralytics results box object."""
    box = MagicMock()
    box.cls = [label_idx]
    box.conf = [conf]
    box.xyxy = [xyxy]
    return box


def _make_result(boxes):
    """Build a fake ultralytics Results object."""
    result = MagicMock()
    result.boxes = boxes
    return result


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestYOLODetector:
    """Tests for services.vision.yolo_detector.YOLODetector."""

    def test_detect_returns_detections(self, tmp_path):
        """Should parse YOLO results into Detection objects correctly."""
        import asyncio
        from services.vision.yolo_detector import YOLODetector, _run_detection

        # Create a dummy frame file so the path exists
        frame = tmp_path / "frame_001.jpg"
        frame.write_bytes(b"\xff\xd8\xff")  # minimal JPEG header

        fake_boxes = [
            _make_box(0, 0.91, [10.0, 20.0, 100.0, 200.0]),  # "person"
            _make_box(63, 0.75, [5.0, 5.0, 50.0, 50.0]),     # "laptop"
        ]
        fake_result = _make_result(fake_boxes)

        fake_model = MagicMock()
        fake_model.names = {0: "person", 63: "laptop"}
        fake_model.predict.return_value = [fake_result]

        with patch("services.vision.yolo_detector._model", fake_model):
            detections = _run_detection([str(frame)], conf_threshold=0.4)

        assert len(detections) == 2
        labels = {d.label for d in detections}
        assert labels == {"person", "laptop"}

    def test_detect_skips_low_confidence(self, tmp_path):
        """Detections below conf_threshold should already be filtered by YOLO predict()."""
        from services.vision.yolo_detector import _run_detection

        frame = tmp_path / "frame_002.jpg"
        frame.write_bytes(b"\xff\xd8\xff")

        # No boxes returned (model filtered them internally)
        fake_result = _make_result(boxes=[])
        fake_model = MagicMock()
        fake_model.names = {}
        fake_model.predict.return_value = [fake_result]

        with patch("services.vision.yolo_detector._model", fake_model):
            detections = _run_detection([str(frame)], conf_threshold=0.7)

        assert detections == []

    def test_detect_empty_frame_list(self):
        """No frames → empty detections without calling model."""
        from services.vision.yolo_detector import _run_detection

        fake_model = MagicMock()
        with patch("services.vision.yolo_detector._model", fake_model):
            detections = _run_detection([], conf_threshold=0.4)

        assert detections == []
        fake_model.predict.assert_not_called()

    def test_detect_handles_frame_error_gracefully(self, tmp_path):
        """If model.predict raises on a frame, it logs a warning and continues."""
        from services.vision.yolo_detector import _run_detection

        frame = tmp_path / "bad_frame.jpg"
        frame.write_bytes(b"not-an-image")

        fake_model = MagicMock()
        fake_model.names = {}
        fake_model.predict.side_effect = RuntimeError("corrupt image")

        with patch("services.vision.yolo_detector._model", fake_model):
            detections = _run_detection([str(frame)], conf_threshold=0.4)

        # Should not raise — graceful degradation
        assert detections == []

    def test_is_available_true_when_model_loaded(self):
        """is_available() returns True when the model singleton is set."""
        from services.vision.yolo_detector import YOLODetector

        detector = YOLODetector()
        fake_model = MagicMock()
        with patch("services.vision.yolo_detector._model", fake_model):
            assert detector.is_available() is True

    def test_is_available_false_when_model_none(self):
        """is_available() returns False when the model failed to load."""
        from services.vision.yolo_detector import YOLODetector

        detector = YOLODetector()
        with patch("services.vision.yolo_detector._model", None):
            with patch("services.vision.yolo_detector._load_model", return_value=None):
                assert detector.is_available() is False
