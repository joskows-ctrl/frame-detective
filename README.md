# Frame Detective V3

Detect and fix missing or duplicate frames in AI-generated video.

By Jonah Oskow

![Frame Detective](https://img.shields.io/badge/version-3.0-blue)

## What It Does

Frame Detective analyzes video frame-by-frame using optical flow to find:
- **Spikes** (red) — Missing frames where motion jumps unnaturally.
- **Dips** (yellow) — Duplicate/near-duplicate frames that cause stuttering.

Interactive chart with zoom, playback controls, and per-frame preview lets you review and fine-tune detections before fixing.

## V3 Changes

- Separate threshold sliders for spikes and dips (adjust without re-analyzing)
- Ignore toggles to disable spike or dip detection independently
- Three output routes: Prep for External Solve, Internal Solve (RIFE AI), Debug
- Output format selection upfront (H.264 or ProRes HQ)
- Dips are replaced (not deleted) — video duration is preserved

## GPU Install (Recommended for Speed)

For NVIDIA GPU acceleration with RIFE AI interpolation:

1. Install [Python 3.10+](https://www.python.org/downloads/) (check "Add to PATH")
2. Clone or download this repo
3. Double-click **`Install_FrameDetective.bat`**
4. Double-click **`Run_FrameDetective.bat`**

The installer automatically detects your GPU and installs CUDA PyTorch. If no NVIDIA GPU is found, it falls back to CPU.

### Manual Install

```bash
python -m venv frame_detective_env
frame_detective_env\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
frame_detective_env\Scripts\pip install opencv-python-headless ccvfi Pillow
frame_detective_env\Scripts\python frame_detective_gui.py
```

For CPU-only:
```bash
frame_detective_env\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Features

- Separate spike and dip threshold sliders with instant re-detection
- Ignore toggles for spikes and dips
- RIFE v4.26 AI frame interpolation (GPU accelerated)
- Three output routes: Prep for External, Internal Solve, Debug
- Interactive motion chart with zoom and moveable cursor
- Video playback with speed control and loop
- In/Out points for selective fixing
- Per-frame review and manual adjustment
- H.264 (25Mbps via FFmpeg) and ProRes HQ output

## Requirements

- Python 3.10+
- FFmpeg (for H.264/ProRes output — optional, falls back to OpenCV)
- NVIDIA GPU with CUDA support (optional, for fast RIFE interpolation)

## License

MIT License
