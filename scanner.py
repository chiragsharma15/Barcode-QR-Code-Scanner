"""
Barcode & QR Code Scanner
=========================
Real-time barcode and QR code scanning using OpenCV and PyZBar.

Features:
  - Live webcam feed with real-time detection
  - Supports: QR Code, EAN, UPC, Code 128, Code 39, PDF417
  - Image preprocessing pipeline (grayscale, histogram eq, Gaussian blur, thresholding)
  - Multi-barcode detection in a single frame
  - Bounding-box visualization with decoded text overlay
  - Scan history (deduplication)
  - FPS optimization (skip every other frame)
  - CSV export of all scanned results

Usage:
  python scanner.py

Controls:
  Q     → Quit
  S     → Save current frame as screenshot
  C     → Clear scan history
  E     → Export history to barcode_data.csv
"""

import cv2
import csv
import os
import datetime
from pyzbar.pyzbar import decode
import numpy as np

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
WINDOW_TITLE      = "Barcode & QR Code Scanner  |  Press Q to quit"
CSV_OUTPUT_FILE   = "barcode_data.csv"
SCREENSHOTS_DIR   = "screenshots"
BOX_COLOR         = (0, 255, 100)       # neon green bounding box
TEXT_COLOR        = (0, 255, 100)       # same colour for label
DUPE_COOLDOWN_SEC = 3                   # seconds before same code can be logged again


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Image preprocessing pipeline:
      1. Grayscale   – barcode detection depends on intensity, not colour
      2. Hist-EQ     – normalises brightness for low-light / uneven conditions
      3. Gaussian    – removes camera noise that confuses edge detection
      4. Threshold   – produces clean black/white for precise boundary detection
    Returns the thresholded image used by PyZBar.
    """
    gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)                         # lighting enhancement
    blurred   = cv2.GaussianBlur(equalized, (5, 5), 0)        # noise removal
    _, thresh = cv2.threshold(                                  # binary threshold
        blurred, 100, 255, cv2.THRESH_BINARY
    )
    return thresh


def draw_overlay(frame: np.ndarray, code, data: str) -> None:
    """Draw bounding box + decoded text above it."""
    x, y, w, h = code.rect

    # Bounding box
    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        BOX_COLOR,
        2
    )

    # Corner accents – makes it look more professional
    corner_len = 14
    thickness  = 3
    for cx, cy, dx, dy in [
        (x,         y,          1,  1),
        (x + w,     y,         -1,  1),
        (x,         y + h,      1, -1),
        (x + w,     y + h,     -1, -1),
    ]:
        cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), BOX_COLOR, thickness)
        cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), BOX_COLOR, thickness)

    # Label background
    label      = f"{code.type}  {data}"
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    label_y    = max(y - 10, th + 10)
    cv2.rectangle(
        frame,
        (x - 1, label_y - th - baseline - 4),
        (x + tw + 4, label_y + baseline),
        (0, 0, 0),
        cv2.FILLED
    )

    # Decoded text
    cv2.putText(
        frame,
        label,
        (x, label_y - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        TEXT_COLOR,
        2,
        cv2.LINE_AA
    )


def draw_hud(
    frame: np.ndarray,
    fps: float,
    count: int,
    history_len: int,
) -> None:
    """Heads-Up Display: FPS + stats bar at the top."""
    h, w = frame.shape[:2]
    bar_h = 36

    # Semi-transparent banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (10, 10, 10), cv2.FILLED)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Codes this frame: {count}", (120, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Total scanned: {history_len}", (330, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 1, cv2.LINE_AA)

    hint = "Q:Quit  S:Screenshot  C:Clear  E:Export"
    cv2.putText(frame, hint, (w - 370, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)


def export_csv(history: list, filepath: str) -> None:
    """Write scan history to a CSV file."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Type", "Value"])
        writer.writerows(history)
    print(f"[EXPORT]  {len(history)} records saved → {filepath}")


def save_screenshot(frame: np.ndarray, directory: str) -> str:
    """Save the current frame as a PNG screenshot."""
    os.makedirs(directory, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(directory, f"scan_{ts}.png")
    cv2.imwrite(path, frame)
    return path


# ─────────────────────────────────────────────
# Main Scanner Loop
# ─────────────────────────────────────────────
def main() -> None:
    # Camera setup – 0 = default webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. "
            "Ensure a camera is connected and not used by another app."
        )

    # Optional: request higher resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    history: list[tuple[str, str, str]] = []          # (timestamp, type, value)
    seen:    dict[str, datetime.datetime] = {}         # last-seen times for dedup

    frame_count = 0
    prev_time   = datetime.datetime.now()
    fps         = 0.0

    print("=" * 55)
    print("  Barcode & QR Code Scanner  –  Real-Time Pipeline")
    print("=" * 55)
    print("  Controls:")
    print("    Q → Quit          S → Screenshot")
    print("    C → Clear history E → Export CSV")
    print("=" * 55)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from webcam.")
            break

        frame_count += 1

        # ── FPS Optimisation: skip every other frame for heavy processing ──
        if frame_count % 2 == 0:
            codes = []
        else:
            processed = preprocess_frame(frame)
            codes      = decode(processed)

        # ── FPS calculation ──
        now      = datetime.datetime.now()
        elapsed  = (now - prev_time).total_seconds()
        if elapsed > 0:
            fps = 1.0 / elapsed
        prev_time = now

        # ── Process each detected code ──
        for code in codes:
            data = code.data.decode("utf-8")

            # Deduplicate: only log if not seen recently
            last_seen = seen.get(data)
            if last_seen is None or (now - last_seen).total_seconds() > DUPE_COOLDOWN_SEC:
                ts_str = now.strftime("%H:%M:%S")
                history.append((ts_str, code.type, data))
                seen[data] = now
                print(f"[SCAN]  {ts_str}  [{code.type}]  {data}")

            draw_overlay(frame, code, data)

        # ── HUD ──
        draw_hud(frame, fps, len(codes), len(history))

        cv2.imshow(WINDOW_TITLE, frame)

        # ── Key handling ──
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:          # Q or ESC → quit
            break
        elif key == ord("s"):                      # S → screenshot
            path = save_screenshot(frame, SCREENSHOTS_DIR)
            print(f"[SCREENSHOT]  Saved → {path}")
        elif key == ord("c"):                      # C → clear history
            history.clear()
            seen.clear()
            print("[CLEAR]  Scan history cleared.")
        elif key == ord("e"):                      # E → export CSV
            export_csv(history, CSV_OUTPUT_FILE)

    # ── Cleanup ──
    cap.release()
    cv2.destroyAllWindows()

    if history:
        print(f"\n[DONE]  {len(history)} unique codes scanned this session.")
        export_csv(history, CSV_OUTPUT_FILE)
    else:
        print("\n[DONE]  No codes were scanned this session.")


if __name__ == "__main__":
    main()
