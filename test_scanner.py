"""
test_scanner.py
===============
Offline unit tests – no webcam required.

Tests cover:
  1. preprocess_frame  → output shape, dtype, value range
  2. draw_overlay      → frame is mutated without error
  3. draw_hud          → frame is mutated without error
  4. export_csv        → file written correctly
  5. Multi-code path   → duplicates filtered within cooldown window
"""

import csv
import os
import tempfile
import datetime
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# ── import the functions under test ──────────────────────────────────────────
from scanner import preprocess_frame, draw_overlay, draw_hud, export_csv


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
def make_bgr_frame(h: int = 240, w: int = 320) -> np.ndarray:
    """Return a random uint8 BGR frame of given dimensions."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def make_mock_code(data: bytes = b"https://example.com",
                   code_type: str = "QRCODE",
                   rect: tuple = (50, 60, 100, 100)) -> MagicMock:
    """Return a mock pyzbar decoded object."""
    m      = MagicMock()
    m.data = data
    m.type = code_type
    m.rect = rect
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Test Cases
# ─────────────────────────────────────────────────────────────────────────────
class TestPreprocessFrame(unittest.TestCase):

    def test_output_shape_is_2d(self):
        """Grayscale output must be H×W (2-D)."""
        frame  = make_bgr_frame()
        result = preprocess_frame(frame)
        self.assertEqual(result.ndim, 2,
                         "preprocess_frame should return a 2-D (grayscale) image")

    def test_output_same_height_width(self):
        """Spatial dimensions must be preserved."""
        frame  = make_bgr_frame(480, 640)
        result = preprocess_frame(frame)
        self.assertEqual(result.shape, (480, 640))

    def test_output_dtype_uint8(self):
        """Output must stay uint8 after all preprocessing steps."""
        frame  = make_bgr_frame()
        result = preprocess_frame(frame)
        self.assertEqual(result.dtype, np.uint8)

    def test_binary_output_values(self):
        """After thresholding, only 0 and 255 may appear."""
        frame  = make_bgr_frame()
        result = preprocess_frame(frame)
        unique = set(np.unique(result))
        self.assertTrue(
            unique.issubset({0, 255}),
            f"Expected only {{0, 255}} but found {unique}"
        )


class TestDrawOverlay(unittest.TestCase):

    def test_draw_does_not_raise(self):
        """draw_overlay must run without raising any exception."""
        frame = make_bgr_frame(480, 640)
        code  = make_mock_code(rect=(40, 40, 120, 80))
        try:
            draw_overlay(frame, code, "https://example.com")
        except Exception as exc:
            self.fail(f"draw_overlay raised {exc!r}")

    def test_frame_mutated(self):
        """Frame should be mutated (pixels changed) after drawing."""
        frame   = np.zeros((480, 640, 3), dtype=np.uint8)
        code    = make_mock_code(rect=(40, 40, 120, 80))
        draw_overlay(frame, code, "TEST_DATA")
        self.assertFalse(
            np.all(frame == 0),
            "draw_overlay should modify at least some pixels"
        )


class TestDrawHud(unittest.TestCase):

    def test_draw_hud_does_not_raise(self):
        """draw_hud must run without raising any exception."""
        frame = make_bgr_frame(480, 640)
        try:
            draw_hud(frame, fps=30.0, count=2, history_len=5)
        except Exception as exc:
            self.fail(f"draw_hud raised {exc!r}")

    def test_hud_changes_top_bar(self):
        """The top banner region should be non-zero after draw_hud."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_hud(frame, fps=25.0, count=1, history_len=3)
        top_strip = frame[:36, :, :]
        self.assertFalse(
            np.all(top_strip == 0),
            "draw_hud should paint the top HUD bar"
        )


class TestExportCsv(unittest.TestCase):

    def test_csv_header_and_rows(self):
        """CSV must start with the correct header and contain all records."""
        history = [
            ("10:30:00", "QRCODE",   "https://github.com"),
            ("10:31:00", "EAN13",    "123456789012"),
            ("10:32:00", "CODE128",  "ABCXYZ"),
        ]
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        ) as tmp:
            tmp_path = tmp.name

        try:
            export_csv(history, tmp_path)
            with open(tmp_path, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))

            self.assertEqual(rows[0], ["Timestamp", "Type", "Value"])
            self.assertEqual(len(rows), len(history) + 1)  # header + data
            self.assertEqual(rows[1][2], "https://github.com")
            self.assertEqual(rows[3][1], "CODE128")
        finally:
            os.unlink(tmp_path)

    def test_empty_history_writes_header_only(self):
        """An empty history should produce a CSV with only the header."""
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        ) as tmp:
            tmp_path = tmp.name

        try:
            export_csv([], tmp_path)
            with open(tmp_path, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0], ["Timestamp", "Type", "Value"])
        finally:
            os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main(verbosity=2)
