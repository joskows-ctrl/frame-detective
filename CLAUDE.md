# Frame Detective V3 — Claude Context

## What This Is
A desktop GUI tool (Python/Tkinter) that detects and fixes missing or duplicate frames in AI-generated video. Built by Jonah Oskow. Website: highlyappropriate.com

## The Problem It Solves
AI video generators (Sora, Runway, Kling, etc.) often produce videos with:
- **Spikes** (missing frames): motion jumps unnaturally between two frames → feels jerky
- **Dips** (duplicate/near-duplicate frames): consecutive frames are nearly identical → feels stuttery

## How It Works
1. **Optical flow analysis**: Computes motion magnitude between every consecutive frame pair using OpenCV's Farneback optical flow
2. **Spike detection**: Flags frames where motion / local_median > spike_threshold (default 2.0). These are frames where too much happened — indicating missing intermediate frames
3. **Dip detection**: Flags frames where motion / local_median < dip_threshold (default 0.70). These are frames where too little happened — indicating duplicated/stuck frames
4. **Visualization**: Interactive chart (red bars = spikes, yellow bars = dips, blue = normal), with zoom, cursor, playback, in/out points
5. **Fixing**: Three output routes handle the flagged frames differently

## Key Architecture
- Single file: `frame_detective_gui.py` (~2800 lines)
- Class: `FrameDetectiveApp` — Tkinter GUI app
- RIFE v4.26 AI interpolation via `ccvfi` package (in `rife/` submodule)
- FFmpeg for H.264/ProRes output (falls back to OpenCV if not found)

## V3 Changes (Current Session)
These are the changes made from V1 → V3:

### Threshold Sliders (No Re-Analysis)
- Spike and dip thresholds are independent sliders
- Adjusting them re-runs detection instantly on cached magnitudes data — no video re-analysis needed
- Spike slider: 0.05 → 5.0 (lower = more sensitive, catches more). At 5.0 auto-checks "Ignore Spikes"
- Dip slider: 0.10 → 0.95 (higher = more sensitive, catches more). At 0.10 auto-checks "Ignore Dips"
- Direction labels: "← more events found" / "less events found →"
- Sliders are extra-wide (500px) for fine control

### Ignore Toggles
- Checkbox for "Ignore Spikes" and "Ignore Dips"
- Auto-checked when slider hits the "off" extreme, auto-unchecked when moved back
- Works across all output routes

### Three Output Routes
1. **Prep for External Solve**: For tools like Topaz Video AI
   - Spikes: duplicates the previous frame and inserts it before the spike frame (external tool interpolates)
   - Dips: replaces the dip frame with a copy of the previous frame
2. **Internal Solve**: Uses RIFE AI interpolation
   - Spikes: inserts RIFE-interpolated frame(s) between previous and spike frame
   - Dips: replaces dip frame with RIFE-interpolated frame between previous and next
3. **Debug**: For visual inspection
   - Spikes: inserts black frame with burned-in label ("INSERT 1/1 @ F42")
   - Dips: replaces with black frame + label ("F42 [DIP]")
   - All frames get burned-in frame numbers (original video frame numbers, not output frame numbers)

### Output Formats
- H.264 (.mp4) — 25Mbps via FFmpeg
- ProRes HQ (.mov) — via FFmpeg
- Both shown upfront as selectable options (no surprises)
- Falls back to OpenCV if FFmpeg not found

### UI Layout (Top to Bottom)
1. Logo + title + website link
2. Load Video button
3. Analyze button (auto-analyzes if video < 2 minutes)
4. Threshold sliders (spike + dip, with ignore checkboxes)
5. Progress bar + status
6. Output section (route selector, format selector, Output button, Export Report)
7. Visualization section (chart + zoom + spike/dip table + frame preview + playback)
8. Full-app scrollbar on right side

### Right-Click Context Menu on Chart
- Right-click any bar on the chart to get: "Set as Spike" / "Set as Dip" / "Disable" for that frame
- Any frame can be set to anything, regardless of its detected type
- Setting type always re-enables the frame (even if previously disabled via In–Out or ✕)
- "Remove from table" was dropped — Disable (set count to 0) covers the use case
- Chart colors update immediately on type change — `dip_entries` is the source of truth, not the auto-detected lists

### Frame Table
- Shows Type, Frame#, Motion, Local Avg, Ratio, Dupes columns
- **Dips**: read-only count label (always 0 or 1) — no manual typing, controlled via ✕ and right-click only
- **Spikes**: editable spinbox (1–10) for adding duplicate frames. ✕ button is the only way to disable (set to 0)
- Disabled rows (count=0) gray out all text in the row for visual clarity
- Mouse wheel scrolls the table (not the spinbox values — spinbox scroll disabled)
- Bulk controls: "Set All to Auto", "Set All to N", "Disable In–Out", "Enable In–Out"
- Set All / bulk buttons are pinned below the scrollable area
- All table labels use tk.Label (not ttk.Label) for consistent styling/graying

### Save/Load Projects
- File > Save / Save As / Open Project (also Ctrl+S, Ctrl+Shift+S, Ctrl+O)
- Saves as `.fdproj` JSON: video path, magnitudes, spikes, dips, user edits, thresholds, settings
- Does NOT save raw frames (too large) — reloads from video file on open
- Warns if video file has moved

### Always Full Precision
- RIFE always runs in full precision (fp32). The half-precision toggle was removed.

### Legacy Code
- `frame_detective_v2.py` — previous version, kept for reference
- `detect_missing_frames.py` — original CLI prototype
- `test_interp.py`, `test_interp_v2.py` — interpolation test scripts
- Recursive fix mode is shelved (UI hidden but code preserved in the class)
- RIFE + Labels fill mode removed from UI but code preserved in class

## Running Locally
```bash
python frame_detective_gui.py
```
Uses system Python (`C:\Users\josko\AppData\Local\Programs\Python\Python313\python`). The `frame_detective_env` venv does NOT exist — all dependencies are installed in system Python. `Run_FrameDetective.bat` will not work.

## Key Methods
- `run_analysis()` — optical flow loop, stores magnitudes + frames_data
- `detect_spikes(magnitudes, threshold)` — ratio vs local median
- `detect_dips(magnitudes, dip_threshold)` — ratio vs local median (excludes spike frames from median)
- `_on_threshold_change()` — re-runs detection on cached data when sliders move
- `show_results()` — builds chart + spike table
- `fix_video()` — main output pipeline, handles all 3 routes
- `interpolate_frame_rife()` — RIFE v4.26 interpolation via ccvfi
- `draw_chart()` — renders the motion magnitude visualization
- `save_project()` / `load_project()` — .fdproj serialization

## Dependencies
- Python 3.10+
- opencv-python-headless (cv2)
- numpy
- Pillow (PIL)
- torch + torchvision (CUDA recommended)
- ccvfi (RIFE wrapper)
- FFmpeg (optional, for H.264/ProRes — falls back to OpenCV)
- tkinter (stdlib)
- tkinterdnd2 (optional, for drag-and-drop)

## Git Remote
- GitHub: joskows-ctrl/frame-detective
- The git version is old (V1). Local code is V3 and ahead of remote.
- README.md has been updated to V3.
