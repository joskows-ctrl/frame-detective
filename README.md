# Frame Detective Build 2

Detect and fix missing or duplicate frames in AI-generated video.

By Jonah Oskow — [highlyappropriate.com](https://highlyappropriate.com)

![Frame Detective](https://img.shields.io/badge/build-2-blue)

## What It Does

Frame Detective analyzes video frame-by-frame using optical flow to find:
- **Spikes** (red) — Missing frames where motion jumps unnaturally.
- **Dips** (yellow) — Duplicate/near-duplicate frames that cause stuttering.

Interactive chart with zoom, playback controls, and per-frame preview lets you review and fine-tune detections before fixing.

## Three Output Routes
1. **Prep for External Solve** — Duplicates frames for tools like Topaz Video AI to interpolate
2. **Internal Solve** — Uses RIFE v4.26 AI interpolation (GPU accelerated)
3. **Debug** — Black frames with burned-in labels for visual inspection

## Features

- Separate spike and dip threshold sliders with instant re-detection
- Ignore toggles for spikes and dips
- Right-click chart bars to reclassify frames (spike ↔ dip ↔ disable)
- User reclassifications persist across threshold changes
- RIFE v4.26 AI frame interpolation (GPU accelerated)
- Interactive motion chart with zoom and moveable cursor
- Video playback with speed control and loop
- In/Out points for selective fixing
- Per-frame review and manual adjustment
- H.264 (25Mbps via FFmpeg) and ProRes HQ output
- Save/Load projects (.fdp files)

## Running from Source

### Install Dependencies
```bash
pip install opencv-python-headless numpy Pillow ccvfi
```

For GPU (NVIDIA CUDA — recommended):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

For CPU only:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Run
```bash
python frame_detective_gui.py
```

### Optional
- Install [FFmpeg](https://ffmpeg.org/) for H.264/ProRes output (falls back to OpenCV if not found)

## Building the EXE

The app can be packaged as a standalone `.exe` (~2.4 GB with GPU support).

### Prerequisites
```bash
pip install pyinstaller
```

### Step 1: Download the RIFE model (required before building)
```bash
python -c "from ccvfi.auto.model import AutoModel; from ccvfi.type import ConfigType; AutoModel.from_pretrained(pretrained_model_name=ConfigType.RIFE_IFNet_v426_heavy)"
```

### Step 2: Build
```bash
pyinstaller FrameDetectiveV4.spec --clean
```

The EXE will be in `dist/`.

### Important Build Notes

**Do NOT exclude any torch modules from the spec file.** Torch's internal import chains are deeply tangled — excluding `torch.distributed`, `torch.testing`, `caffe2`, etc. will cause runtime crashes. The `excludes` list must stay empty.

**RIFE model weights are bundled in the EXE.** The `ccvfi` package normally downloads model weights on first use, but this fails in a windowless EXE (no console for the download progress bar). The spec file bundles the `.pkl` model file, and the app code points to `sys._MEIPASS/cache_models/` when running frozen.

**Kill any running EXE before rebuilding.** PyInstaller can't overwrite a locked file. The EXE may spawn child processes that survive window close — use Task Manager if needed.

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (optional, for fast RIFE interpolation)
- FFmpeg (optional, for H.264/ProRes output)

## License

MIT License
