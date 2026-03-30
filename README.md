# Frame Detective

Detect and fix missing or duplicate frames in AI-generated video.

![Frame Detective](https://img.shields.io/badge/build-3-blue)

## What It Does

AI video generators (Sora, Runway, Kling, Seedance, etc.) often produce videos with temporal artifacts:
- **Spikes** (red) — Missing frames where motion jumps unnaturally between two frames
- **Dips** (yellow) — Duplicate/near-duplicate frames that cause stuttering

Frame Detective uses optical flow analysis to detect these artifacts, then lets you review, adjust, and export corrected timelines.

## Features

- Optical flow analysis with separate spike and dip threshold sliders
- Interactive motion chart with zoom, cursor, and per-frame video preview
- Right-click chart bars to reclassify frames (spike ↔ dip ↔ disable)
- User reclassifications persist across threshold changes
- In/Out points with bulk enable/disable
- Video playback with speed control and loop
- Keyboard shortcuts: I/O (in/out), Space (play), Arrow keys (step)
- Drag-and-drop video loading
- Auto-analyze for videos under 1 minute
- Stop analysis mid-way and keep partial results
- Disk cache mode for long videos (saves RAM)
- Save/Load projects (.fdp files)

## Two Output Modes

### Prep
Inserts a black frame before each spike frame. Use this to prepare footage for external interpolation tools (Topaz Video AI, etc.).

### Map
All-black video matching original length. Spike frames show "SPIKE", dip frames show "DIP", with velocity data burned in. Useful for visual reference alongside the original.

### FCP7 XML Export
Export a Final Cut Pro 7 XML timeline with clip segments and sequence markers (red for spikes, yellow for dips). Works in Premiere Pro, DaVinci Resolve, and other NLEs that import FCP7 XML. Can be exported standalone (offline reference) or alongside Prep/Map output.

## Running from Source

### Install Dependencies
```bash
pip install opencv-python-headless numpy Pillow
```

Optional for drag-and-drop:
```bash
pip install tkinterdnd2
```

### Run
```bash
python frame_detective_v3.py
```

### Optional
- Install [FFmpeg](https://ffmpeg.org/) for H.264 output (falls back to OpenCV if not found)

## Building the EXE

```bash
pip install pyinstaller
pyinstaller FrameDetectiveV3.spec --clean
```

Output: `dist/FrameDetective_V3/` (~164 MB). No GPU or CUDA required.

## Requirements

- Python 3.10+
- FFmpeg (optional, for H.264 output)

## License

MIT License
