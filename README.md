# Barcode & QR Code Scanner

> Real-time computer-vision pipeline — Python · OpenCV · PyZBar

---

## Architecture

```
Webcam Feed
     ↓
Frame Capture      (cv2.VideoCapture)
     ↓
Image Preprocessing
  ├─ Grayscale         (COLOR_BGR2GRAY)
  ├─ Histogram EQ      (equalizeHist)
  ├─ Gaussian Blur     (5×5 kernel)
  └─ Binary Threshold  (THRESH_BINARY)
     ↓
Barcode Detection  (pyzbar.decode)
     ↓
Decoding + Dedup
     ↓
Overlay + HUD
     ↓
Display / Export
```

---

## Supported Formats

| Format   | Example value         |
|----------|-----------------------|
| QR Code  | https://github.com    |
| EAN-13   | 5901234123457         |
| UPC-A    | 012345678905          |
| Code 128 | ABCXYZ-001            |
| Code 39  | HELLO                 |
| PDF417   | (multi-line payload)  |

---

## Setup

```bash
# 1. Clone / open the project folder
# 2. Install dependencies
pip install -r requirements.txt
```

> **Windows only** – if you get a `zbar` DLL error, install the
> [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
> or place the `libzbar-64.dll` next to your Python executable.

---

## Run

```bash
python scanner.py
```

### Controls

| Key | Action             |
|-----|--------------------|
| `Q` / `ESC` | Quit         |
| `S` | Save screenshot    |
| `C` | Clear scan history |
| `E` | Export CSV now     |

---

## Run Tests (no webcam needed)

```bash
python -m pytest test_scanner.py -v
# or
python test_scanner.py
```

---

## Project Structure

```
Barcode & QR code scanner/
├── scanner.py          # Main application
├── test_scanner.py     # Offline unit tests
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── barcode_data.csv    # Auto-generated on exit / E key
└── screenshots/        # Auto-created on S key
```

---

## Resume Bullet Points

```
Barcode & QR Code Scanner | Python, OpenCV, PyZBar

• Developed a real-time barcode and QR code scanning system using OpenCV and PyZBar.
• Implemented image preprocessing techniques including grayscale conversion,
  histogram equalization, and Gaussian noise reduction to improve detection accuracy
  under varying lighting conditions.
• Enabled real-time decoding and visualization of multiple barcode formats from live
  webcam feeds with bounding-box and corner-accent overlays.
• Designed a low-latency processing pipeline (FPS-optimized frame skipping) supporting
  EAN, UPC, Code 128, Code 39, and QR Code standards.
• Built automated scan-history deduplication and CSV export for inventory tracking use-cases.
• Wrote offline unit tests covering the preprocessing, rendering, and I/O layers.
```
