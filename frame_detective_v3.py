"""
Frame Detective V3 — Detect spikes & dips in AI-generated video
Simplified rebuild: no interpolation, no table column.
Everything happens in the visualizer. Output: Prep video, Map video, FCP7 XML.
By Jonah Oskow — highlyappropriate.com
"""

import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading
import subprocess
import shutil
import webbrowser
import os
import json
import tempfile
import xml.etree.ElementTree as ET
from xml.dom import minidom
from urllib.parse import quote as url_quote

# Hide console window for subprocesses on Windows
_SUBPROCESS_FLAGS = 0
if sys.platform == "win32":
    _SUBPROCESS_FLAGS = subprocess.CREATE_NO_WINDOW


class ToolTip:
    """Lightweight tooltip for Tkinter widgets."""
    def __init__(self, widget, text, delay=500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._tip_window = None
        self._after_id = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._cancel, add="+")
        widget.bind("<ButtonPress>", self._cancel, add="+")

    def _schedule(self, event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel(self, event=None):
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None
        self._hide()

    def _show(self):
        if self._tip_window:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, bg="#333333", fg="#eeeeee",
                         font=("Segoe UI", 9), padx=6, pady=3, relief=tk.SOLID, borderwidth=1)
        label.pack()

    def _hide(self):
        if self._tip_window:
            self._tip_window.destroy()
            self._tip_window = None


class FrameDetectiveApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Frame Detective V3")
        self.root.geometry("1100x820")
        self.root.configure(bg="#1a1a1a")
        self.root.resizable(True, True)

        # Set window icon
        try:
            if getattr(sys, 'frozen', False):
                icon_path = Path(sys._MEIPASS) / "fd_icon.ico"
            else:
                icon_path = Path(__file__).parent / "fd_icon.ico"
            if icon_path.exists():
                self.root.iconbitmap(str(icon_path))
        except Exception:
            pass

        # Core state
        self.video_path = None
        self.magnitudes = []       # list of (frame_idx, avg_motion_mag)
        self.frames_data = []      # list of (frame_idx, ndarray) — only used when disk cache is OFF
        self.spikes = []           # auto-detected spikes
        self.dips = []             # auto-detected dips
        self.fps = 0
        self.width = 0
        self.height = 0
        self.total_frames = 0
        self.analyzing = False
        self._stop_analysis = False

        # Disk cache state
        self._disk_cache_dir = None  # temp dir path when disk caching is active
        self._disk_cache_count = 0   # number of frames cached to disk

        # User overrides: {frame_num: {"type": "spike"|"dip", "enabled": bool}}
        self.user_overrides = {}

        # Chart / cursor state
        self._cursor_frame_idx = None
        self._cursor_locked = False
        self._cursor_line = None
        self._chart_geom = None
        self._in_point = None
        self._out_point = None
        self._in_line = None
        self._out_line = None
        self._zoom_start = 0
        self._zoom_end = None
        self._zoom_drag_mode = None

        # Playback state
        self._playing = False
        self._play_after_id = None

        # Guard flag to prevent threshold trace from re-drawing during other operations
        self._in_draw = False

        # Project
        self._project_path = None

        self.setup_styles()
        self.build_ui()

    # ── Styles ────────────────────────────────────────────────────────

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TFrame", background="#1a1a1a")
        style.configure("Dark.TLabel", background="#1a1a1a", foreground="#e0e0e0",
                         font=("Segoe UI", 10))
        style.configure("Title.TLabel", background="#1a1a1a", foreground="#ffffff",
                         font=("Segoe UI", 16, "bold"))
        style.configure("Info.TLabel", background="#1a1a1a", foreground="#888888",
                         font=("Segoe UI", 9))
        style.configure("Dark.TButton", font=("Segoe UI", 10))
        style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"))
        style.configure("Dark.TCheckbutton", background="#1a1a1a", foreground="#e0e0e0",
                         font=("Segoe UI", 9))

    # ── UI Layout ─────────────────────────────────────────────────────

    def build_ui(self):
        main = ttk.Frame(self.root, style="Dark.TFrame", padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        # ── Row 0: Title + File ──
        top = ttk.Frame(main, style="Dark.TFrame")
        top.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(top, text="Frame Detective V3", style="Title.TLabel").pack(side=tk.LEFT)

        # Website link
        link = tk.Label(top, text="highlyappropriate.com", fg="#4a9eff", bg="#1a1a1a",
                        font=("Segoe UI", 9, "underline"), cursor="hand2")
        link.pack(side=tk.LEFT, padx=(12, 0))
        link.bind("<Button-1>", lambda e: webbrowser.open("https://highlyappropriate.com"))

        self.stop_btn = ttk.Button(top, text="Stop", command=self.stop_analyze,
                                    style="Dark.TButton", state=tk.DISABLED)
        self.stop_btn.pack(side=tk.RIGHT, padx=(4, 0))
        ToolTip(self.stop_btn, "Stop analysis early — keeps progress so far")

        self.analyze_btn = ttk.Button(top, text="Analyze", command=self.start_analyze,
                                       style="Accent.TButton")
        self.analyze_btn.pack(side=tk.RIGHT, padx=(4, 0))
        ToolTip(self.analyze_btn, "Run optical flow analysis on the video")

        browse_btn = ttk.Button(top, text="Browse", command=self.browse_file,
                   style="Dark.TButton")
        browse_btn.pack(side=tk.RIGHT, padx=(4, 0))
        ToolTip(browse_btn, "Select a video file (or drag and drop)")

        self.file_label = ttk.Label(top, text="No file selected", style="Info.TLabel")
        self.file_label.pack(side=tk.RIGHT, padx=(8, 4))

        # ── Row 1: Thresholds ──
        thresh_frame = ttk.Frame(main, style="Dark.TFrame")
        thresh_frame.pack(fill=tk.X, pady=(0, 6))

        # Spike threshold
        ttk.Label(thresh_frame, text="Spike:", style="Dark.TLabel").pack(side=tk.LEFT)
        self.spike_threshold_var = tk.DoubleVar(value=2.0)
        self.spike_slider = tk.Scale(thresh_frame, from_=1.05, to=5.0, resolution=0.05,
                                      orient=tk.HORIZONTAL, variable=self.spike_threshold_var,
                                      length=140, bg="#1a1a1a", fg="#e0e0e0",
                                      troughcolor="#333333", highlightthickness=0,
                                      showvalue=True, font=("Consolas", 8))
        self.spike_slider.pack(side=tk.LEFT, padx=(2, 4))
        self.spike_slider.bind("<ButtonRelease-1>", self._on_slider_release)
        self.spike_threshold_var.trace_add("write", self._on_threshold_drag)
        ToolTip(self.spike_slider, "Higher = fewer spikes detected\nMotion must exceed this × local median")

        self.ignore_spikes_var = tk.BooleanVar(value=False)
        ignore_spk = ttk.Checkbutton(thresh_frame, text="Ignore Spikes", variable=self.ignore_spikes_var,
                         style="Dark.TCheckbutton",
                         command=self._on_ignore_toggle)
        ignore_spk.pack(side=tk.LEFT, padx=(0, 16))
        ToolTip(ignore_spk, "Hide all spike markers from the chart")

        # Dip threshold
        ttk.Label(thresh_frame, text="Dip:", style="Dark.TLabel").pack(side=tk.LEFT)
        self.dip_threshold_var = tk.DoubleVar(value=0.70)
        self.dip_slider = tk.Scale(thresh_frame, from_=0.05, to=0.99, resolution=0.01,
                                    orient=tk.HORIZONTAL, variable=self.dip_threshold_var,
                                    length=140, bg="#1a1a1a", fg="#e0e0e0",
                                    troughcolor="#333333", highlightthickness=0,
                                    showvalue=True, font=("Consolas", 8))
        self.dip_slider.pack(side=tk.LEFT, padx=(2, 4))
        self.dip_slider.bind("<ButtonRelease-1>", self._on_slider_release)
        self.dip_threshold_var.trace_add("write", self._on_threshold_drag)
        ToolTip(self.dip_slider, "Lower = fewer dips detected\nMotion must be below this × local median")

        self.ignore_dips_var = tk.BooleanVar(value=False)
        ignore_dip = ttk.Checkbutton(thresh_frame, text="Ignore Dips", variable=self.ignore_dips_var,
                         style="Dark.TCheckbutton",
                         command=self._on_ignore_toggle)
        ignore_dip.pack(side=tk.LEFT)
        ToolTip(ignore_dip, "Hide all dip markers from the chart")

        # Disk cache option (right side of threshold row)
        self.disk_cache_var = tk.BooleanVar(value=False)
        dc_cb = ttk.Checkbutton(thresh_frame, text="Disk Cache", variable=self.disk_cache_var,
                         style="Dark.TCheckbutton")
        dc_cb.pack(side=tk.RIGHT)
        ToolTip(dc_cb, "Save frames to disk instead of RAM\nEnable before analyzing long videos")
        dc_hint = tk.Label(thresh_frame, text="(for long videos)", fg="#666666", bg="#1a1a1a",
                           font=("Segoe UI", 8))
        dc_hint.pack(side=tk.RIGHT, padx=(0, 4))

        # ── Row 2: Chart + Preview (side by side) ──
        viz_frame = ttk.Frame(main, style="Dark.TFrame")
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        # Left: chart + zoom + controls
        left = ttk.Frame(viz_frame, style="Dark.TFrame")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(left, bg="#222222", height=180, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Motion>", self._on_chart_hover)
        self.canvas.bind("<Button-1>", self._on_chart_click)
        self.canvas.bind("<Button-3>", self._on_chart_right_click)
        self.canvas.bind("<MouseWheel>", self._on_chart_scroll)
        self.canvas.bind("<Configure>", lambda e: self.draw_chart())

        # Zoom slider
        self.zoom_slider = tk.Canvas(left, bg="#1a1a1a", height=24, highlightthickness=0)
        self.zoom_slider.pack(fill=tk.X, pady=(2, 4))
        self.zoom_slider.bind("<ButtonPress-1>", self._zoom_slider_press)
        self.zoom_slider.bind("<B1-Motion>", self._zoom_slider_drag)
        self.zoom_slider.bind("<ButtonRelease-1>", self._zoom_slider_release)
        self.zoom_slider.bind("<Double-Button-1>", lambda e: self._zoom_fit())

        # Playback + in/out controls
        ctrl = ttk.Frame(left, style="Dark.TFrame")
        ctrl.pack(fill=tk.X, pady=(0, 2))

        self.play_btn = ttk.Button(ctrl, text="\u25b6 Play", command=self._toggle_play,
                                    style="Dark.TButton", width=8)
        self.play_btn.pack(side=tk.LEFT, padx=(0, 4))
        ToolTip(self.play_btn, "Play / Pause  (Space)")

        self.speed_var = tk.StringVar(value="1x")
        speed_menu = ttk.OptionMenu(ctrl, self.speed_var, "1x", "0.25x", "0.5x", "1x", "2x")
        speed_menu.pack(side=tk.LEFT, padx=(0, 4))
        ToolTip(speed_menu, "Playback speed")

        self.loop_var = tk.BooleanVar(value=True)
        loop_cb = ttk.Checkbutton(ctrl, text="Loop", variable=self.loop_var,
                         style="Dark.TCheckbutton")
        loop_cb.pack(side=tk.LEFT, padx=(0, 8))
        ToolTip(loop_cb, "Loop playback between In/Out or start/end")

        # Step buttons
        step_back10 = ttk.Button(ctrl, text="\u25c0\u25c0", width=3,
                   command=lambda: self._step_frame(-10))
        step_back10.pack(side=tk.LEFT)
        ToolTip(step_back10, "Back 10 frames  (Shift+Left)")

        step_back1 = ttk.Button(ctrl, text="\u25c0", width=3,
                   command=lambda: self._step_frame(-1))
        step_back1.pack(side=tk.LEFT)
        ToolTip(step_back1, "Back 1 frame  (Left)")

        step_fwd1 = ttk.Button(ctrl, text="\u25b6", width=3,
                   command=lambda: self._step_frame(1))
        step_fwd1.pack(side=tk.LEFT)
        ToolTip(step_fwd1, "Forward 1 frame  (Right)")

        step_fwd10 = ttk.Button(ctrl, text="\u25b6\u25b6", width=3,
                   command=lambda: self._step_frame(10))
        step_fwd10.pack(side=tk.LEFT, padx=(0, 8))
        ToolTip(step_fwd10, "Forward 10 frames  (Shift+Right)")

        # In/Out
        in_btn = ttk.Button(ctrl, text="In", width=3,
                   command=self._set_in_point)
        in_btn.pack(side=tk.LEFT, padx=(0, 2))
        ToolTip(in_btn, "Set In point at cursor  (I)")

        out_btn = ttk.Button(ctrl, text="Out", width=3,
                   command=self._set_out_point)
        out_btn.pack(side=tk.LEFT, padx=(0, 2))
        ToolTip(out_btn, "Set Out point at cursor  (O)")

        clr_btn = ttk.Button(ctrl, text="Clr", width=3,
                   command=self._clear_inout)
        clr_btn.pack(side=tk.LEFT, padx=(0, 4))
        ToolTip(clr_btn, "Clear In and Out points")

        self.inout_label = ttk.Label(ctrl, text="In: \u2014  |  Out: \u2014", style="Info.TLabel")
        self.inout_label.pack(side=tk.LEFT, padx=(4, 8))

        disable_btn = ttk.Button(ctrl, text="Disable Range", width=13,
                   command=self._disable_inout_range)
        disable_btn.pack(side=tk.LEFT, padx=(0, 2))
        ToolTip(disable_btn, "Disable all flagged frames in In/Out range\nDisabled frames are excluded from output")

        enable_btn = ttk.Button(ctrl, text="Enable Range", width=12,
                   command=self._enable_inout_range)
        enable_btn.pack(side=tk.LEFT)
        ToolTip(enable_btn, "Re-enable all flagged frames in In/Out range")

        # Right: preview
        right = ttk.Frame(viz_frame, style="Dark.TFrame", width=340)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        right.pack_propagate(False)

        self.preview_label = ttk.Label(right, text="Frame Preview", style="Info.TLabel")
        self.preview_label.pack(pady=(0, 4))

        self.preview_canvas = tk.Canvas(right, bg="#111111", width=320, height=320,
                                         highlightthickness=0)
        self.preview_canvas.pack()
        self._preview_photo = None

        # ── Row 3: Output controls ──
        out_frame = ttk.Frame(main, style="Dark.TFrame")
        out_frame.pack(fill=tk.X, pady=(4, 4))

        ttk.Label(out_frame, text="Format:", style="Dark.TLabel").pack(side=tk.LEFT)
        self.output_format_var = tk.StringVar(value="H.264 (.mp4)")
        fmt_menu = ttk.OptionMenu(out_frame, self.output_format_var,
                                   "H.264 (.mp4)", "H.264 (.mp4)", "ProRes HQ (.mov)")
        fmt_menu.pack(side=tk.LEFT, padx=(2, 12))

        self.prep_btn = ttk.Button(out_frame, text="Export Prep", command=self.export_prep,
                                    style="Accent.TButton")
        self.prep_btn.pack(side=tk.LEFT, padx=(0, 4))
        ToolTip(self.prep_btn, "Insert black frames before spikes\nFor fixing in external tools like Topaz")

        self.map_btn = ttk.Button(out_frame, text="Export Map", command=self.export_map,
                                   style="Accent.TButton")
        self.map_btn.pack(side=tk.LEFT, padx=(0, 4))
        ToolTip(self.map_btn, "Black video with SPIKE/DIP labels\nMatches original frame count")

        xml_btn = ttk.Button(out_frame, text="Export XML Only", command=self.export_xml_only,
                   style="Dark.TButton")
        xml_btn.pack(side=tk.LEFT, padx=(0, 12))
        ToolTip(xml_btn, "FCP7 XML with markers and clip segments\nImport into Premiere Pro or Final Cut")

        # FFmpeg status
        ffmpeg_path = self._find_ffmpeg()
        ff_text = "FFmpeg: found" if ffmpeg_path else "FFmpeg: not found (video-only output)"
        ff_color = "#44aa44" if ffmpeg_path else "#aa4444"
        self._ffmpeg_label = tk.Label(out_frame, text=ff_text, fg=ff_color, bg="#1a1a1a",
                                       font=("Segoe UI", 8))
        self._ffmpeg_label.pack(side=tk.LEFT, padx=(4, 0))

        # Save/Load on right side
        load_btn = ttk.Button(out_frame, text="Load", command=self.open_project,
                   style="Dark.TButton")
        load_btn.pack(side=tk.RIGHT, padx=(2, 0))
        ToolTip(load_btn, "Open a saved .fdp project file")

        save_btn = ttk.Button(out_frame, text="Save", command=self.save_project,
                   style="Dark.TButton")
        save_btn.pack(side=tk.RIGHT, padx=(2, 0))
        ToolTip(save_btn, "Save project (thresholds, overrides, analysis)")

        saveas_btn = ttk.Button(out_frame, text="Save As", command=self.save_project_as,
                   style="Dark.TButton")
        saveas_btn.pack(side=tk.RIGHT, padx=(2, 0))
        ToolTip(saveas_btn, "Save project to a new file")

        # ── Row 4: Progress + Log ──
        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(main, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, pady=(2, 2))

        self.status_label = ttk.Label(main, text="Ready", style="Info.TLabel")
        self.status_label.pack(fill=tk.X)

        self.log = tk.Text(main, height=5, bg="#111111", fg="#cccccc",
                            font=("Consolas", 9), insertbackground="#ffffff",
                            wrap=tk.WORD, state=tk.NORMAL)
        self.log.pack(fill=tk.X, pady=(2, 0))
        self.log.tag_configure("header", foreground="#4a9eff", font=("Consolas", 9, "bold"))
        self.log.tag_configure("good", foreground="#44cc44")
        self.log.tag_configure("error", foreground="#ff4444")
        self.log.tag_configure("warn", foreground="#ffaa44")
        self.log.tag_configure("link", foreground="#4a9eff", underline=True)

        # ── Keyboard shortcuts ──
        self.root.bind("<i>", lambda e: self._set_in_point())
        self.root.bind("<I>", lambda e: self._set_in_point())
        self.root.bind("<o>", lambda e: self._set_out_point())
        self.root.bind("<O>", lambda e: self._set_out_point())
        self.root.bind("<space>", lambda e: self._toggle_play())
        self.root.bind("<Left>", lambda e: self._step_frame(-1))
        self.root.bind("<Right>", lambda e: self._step_frame(1))
        self.root.bind("<Shift-Left>", lambda e: self._step_frame(-10))
        self.root.bind("<Shift-Right>", lambda e: self._step_frame(10))

        # ── Drag-and-drop support ──
        try:
            from tkinterdnd2 import DND_FILES
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:
            pass  # tkinterdnd2 not available, skip drag-and-drop

    def _on_drop(self, event):
        """Handle drag-and-drop of video files."""
        path = event.data.strip()
        # tkinterdnd2 wraps paths with spaces in braces: {C:/path with spaces/file.mp4}
        if path.startswith("{") and path.endswith("}"):
            path = path[1:-1]
        video_exts = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".mxf", ".ts")
        if path.lower().endswith(video_exts):
            self.video_path = Path(path)
            self.file_label.configure(text=self.video_path.name)
            # Auto-analyze if under 1 minute
            try:
                cap = cv2.VideoCapture(str(self.video_path))
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    cap.release()
                    if fps > 0 and total / fps < 60:
                        self.start_analyze()
            except Exception:
                pass
        else:
            messagebox.showwarning("Unsupported file", "Please drop a video file (.mp4, .mov, .avi, .mkv, .webm, .mxf, .ts)")

    # ── Disk Cache ────────────────────────────────────────────────────

    def _using_disk_cache(self):
        return self._disk_cache_dir is not None

    def _init_disk_cache(self):
        """Create temp directory for frame caching."""
        self._disk_cache_dir = tempfile.mkdtemp(prefix="fd_cache_")
        self._disk_cache_count = 0

    def _store_frame(self, frame_idx, frame):
        """Store a frame — to disk cache or RAM depending on mode."""
        if self._using_disk_cache():
            path = os.path.join(self._disk_cache_dir, f"{frame_idx:06d}.bmp")
            cv2.imwrite(path, frame)
            self._disk_cache_count += 1
        else:
            self.frames_data.append((frame_idx, frame.copy()))

    def _get_frame(self, data_idx):
        """Retrieve a frame by its index in magnitudes. Returns (frame_idx, bgr_ndarray) or None."""
        if not self.magnitudes:
            return None
        if data_idx < 0 or data_idx >= len(self.magnitudes):
            return None

        frame_idx = self.magnitudes[data_idx][0]

        if self._using_disk_cache():
            path = os.path.join(self._disk_cache_dir, f"{frame_idx:06d}.bmp")
            if os.path.exists(path):
                frame = cv2.imread(path)
                return (frame_idx, frame) if frame is not None else None
            return None
        else:
            if data_idx < len(self.frames_data):
                return self.frames_data[data_idx]
            return None

    def _get_frame_by_num(self, frame_num):
        """Retrieve a frame by its frame number. Returns bgr_ndarray or None."""
        if self._using_disk_cache():
            path = os.path.join(self._disk_cache_dir, f"{frame_num:06d}.bmp")
            if os.path.exists(path):
                return cv2.imread(path)
            return None
        else:
            for fidx, frame in self.frames_data:
                if fidx == frame_num:
                    return frame
            return None

    def _has_frames(self):
        """Check whether frame data is available (RAM or disk)."""
        if self._using_disk_cache():
            return self._disk_cache_count > 0
        return len(self.frames_data) > 0

    def _frame_count(self):
        """Number of frames stored."""
        if self._using_disk_cache():
            return self._disk_cache_count
        return len(self.frames_data)

    def _cleanup_disk_cache(self):
        """Remove temp directory and all cached frames."""
        if self._disk_cache_dir and os.path.isdir(self._disk_cache_dir):
            try:
                shutil.rmtree(self._disk_cache_dir)
            except Exception:
                pass
        self._disk_cache_dir = None
        self._disk_cache_count = 0

    def _iter_frames(self):
        """Iterate all frames in order. Yields (frame_idx, bgr_ndarray)."""
        if self._using_disk_cache():
            for fidx, _ in self.magnitudes:
                path = os.path.join(self._disk_cache_dir, f"{fidx:06d}.bmp")
                if os.path.exists(path):
                    frame = cv2.imread(path)
                    if frame is not None:
                        yield (fidx, frame)
        else:
            yield from self.frames_data

    # ── Utility ───────────────────────────────────────────────────────

    def log_msg(self, msg, tag=None):
        self.log.insert(tk.END, msg, (tag,) if tag else ())
        self.log.see(tk.END)

    def set_status(self, msg):
        self.status_label.configure(text=msg)

    def _find_ffmpeg(self):
        p = shutil.which("ffmpeg")
        if p:
            return p
        for candidate in [r"C:\ffmpeg\bin\ffmpeg.exe",
                          r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
                          os.path.expanduser(r"~\ffmpeg\bin\ffmpeg.exe")]:
            if os.path.isfile(candidate):
                return candidate
        return None

    def _open_in_explorer(self, filepath):
        filepath = str(filepath)
        if sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", filepath], creationflags=_SUBPROCESS_FLAGS)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", filepath])
        else:
            subprocess.Popen(["xdg-open", os.path.dirname(filepath)])

    def _log_clickable_path(self, filepath):
        tag_name = f"path_{id(filepath)}"
        self.log.tag_configure(tag_name, foreground="#4a9eff", underline=True)
        self.log.tag_bind(tag_name, "<Button-1>",
                          lambda e, p=filepath: self._open_in_explorer(p))
        self.log.insert(tk.END, f"  {filepath}\n", (tag_name,))
        self.log.see(tk.END)

    # ── File Loading & Analysis ───────────────────────────────────────

    def browse_file(self):
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.webm *.mxf *.ts"),
                       ("All files", "*.*")]
        )
        if path:
            self.video_path = Path(path)
            self.file_label.configure(text=self.video_path.name)

            # Auto-analyze if video is under 1 minute
            try:
                cap = cv2.VideoCapture(str(self.video_path))
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    cap.release()
                    if fps > 0 and total / fps < 60:
                        self.start_analyze()
            except Exception:
                pass

    def start_analyze(self):
        if not self.video_path:
            messagebox.showwarning("No file", "Please select a video file first.")
            return
        if self.analyzing:
            return

        self.analyzing = True
        self._stop_analysis = False
        self.analyze_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.canvas.delete("all")

        # Reset state
        self._in_point = None
        self._out_point = None
        self._cursor_frame_idx = None
        self._cursor_locked = False
        self._zoom_start = 0
        self._zoom_end = None
        self.user_overrides.clear()

        # Clean up any previous disk cache
        self._cleanup_disk_cache()
        self.frames_data = []

        # Init disk cache if enabled
        if self.disk_cache_var.get():
            self._init_disk_cache()

        thread = threading.Thread(target=self.run_analysis, daemon=True)
        thread.start()

    def stop_analyze(self):
        """Signal the analysis loop to stop, keeping partial results."""
        self._stop_analysis = True
        self.set_status("Stopping analysis...")

    def run_analysis(self):
        try:
            self.root.after(0, lambda: self.set_status("Opening video..."))

            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                self.root.after(0, lambda: messagebox.showerror("Error", "Cannot open video"))
                return

            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.total_frames = total

            info = f"Video: {self.width}x{self.height} @ {self.fps:.2f}fps, {total} frames\n"
            self.root.after(0, lambda: self.log_msg(info, "header"))

            ret, prev_frame = cap.read()
            if not ret:
                return

            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            self.magnitudes = [(0, 0.0)]  # Frame 0: no motion data, but visible in chart
            self._store_frame(0, prev_frame)

            frame_idx = 1
            stopped_early = False
            while True:
                if self._stop_analysis:
                    stopped_early = True
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                avg_mag = float(np.mean(mag))
                self.magnitudes.append((frame_idx, avg_mag))
                self._store_frame(frame_idx, frame)

                pct = (frame_idx / total) * 100
                self.root.after(0, lambda p=pct, f=frame_idx: (
                    self.progress_var.set(p),
                    self.set_status(f"Analyzing frame {f}/{total}...")
                ))

                prev_gray = gray
                frame_idx += 1

            cap.release()

            # Detect spikes and dips on whatever frames we have
            spike_thresh = self.spike_threshold_var.get()
            dip_thresh = self.dip_threshold_var.get()
            self.spikes = self.detect_spikes(self.magnitudes, spike_thresh)
            self.dips = self.detect_dips(self.magnitudes, dip_thresh)
            self.user_overrides = {}

            n_spikes = len(self.spikes)
            n_dips = len(self.dips)
            analyzed = self._frame_count()
            status = f"Stopped at frame {analyzed}/{total}" if stopped_early else "Analysis complete"
            self.root.after(0, lambda: (
                self.log_msg(f"{'Partial: ' if stopped_early else ''}Found {n_spikes} spikes, {n_dips} dips ({analyzed} frames)\n", "good"),
                self.progress_var.set(100),
                self.set_status(status),
                self.draw_chart()
            ))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, lambda: (
                self.analyze_btn.configure(state=tk.NORMAL),
                self.stop_btn.configure(state=tk.DISABLED),
                setattr(self, 'analyzing', False),
                setattr(self, '_stop_analysis', False)
            ))

    # ── Detection ─────────────────────────────────────────────────────

    def detect_spikes(self, magnitudes, threshold_multiplier=2.0):
        if len(magnitudes) < 5:
            return []

        mags = np.array([m[1] for m in magnitudes])
        window = min(15, len(mags) // 2)
        if window < 3:
            window = 3

        spikes = []
        for i in range(len(mags)):
            start = max(0, i - window)
            end = min(len(mags), i + window + 1)
            local = np.concatenate([mags[start:i], mags[i+1:end]])
            if len(local) == 0:
                continue

            local_median = float(np.median(local))
            local_std = float(np.std(local))

            if local_median > 0:
                ratio = mags[i] / local_median
                if ratio > threshold_multiplier and mags[i] > local_median + local_std:
                    spikes.append({
                        'frame': magnitudes[i][0],
                        'magnitude': float(mags[i]),
                        'local_median': local_median,
                        'ratio': float(ratio),
                    })
        return spikes

    def detect_dips(self, magnitudes, dip_threshold=0.70):
        if len(magnitudes) < 5:
            return []

        mags = np.array([m[1] for m in magnitudes])
        window = min(7, len(mags) // 2)
        if window < 3:
            window = 3

        dips = []
        spike_frame_set = set(s['frame'] for s in self.spikes) if self.spikes else set()

        for i in range(1, len(mags)):
            start = max(0, i - window)
            end = min(len(mags), i + window + 1)
            neighbors = []
            for j in range(start, end):
                if j != i and magnitudes[j][0] not in spike_frame_set:
                    neighbors.append(mags[j])
            if not neighbors:
                local = np.concatenate([mags[start:i], mags[i+1:end]])
            else:
                local = np.array(neighbors)
            if len(local) == 0:
                continue

            local_median = float(np.median(local))

            if local_median > 0:
                ratio = mags[i] / local_median
                if ratio < dip_threshold:
                    dips.append({
                        'frame': magnitudes[i][0],
                        'magnitude': float(mags[i]),
                        'local_median': local_median,
                        'ratio': float(ratio),
                    })
        return dips

    # ── Ground Truth ──────────────────────────────────────────────────

    def _get_ground_truth(self):
        """Build frame classification from auto-detection + user overrides.

        Returns dict: frame_num -> {"type": "spike"|"dip", "enabled": bool,
                                     "source": "auto"|"user",
                                     "magnitude": float, "local_median": float, "ratio": float}
        """
        result = {}

        # Auto-detected spikes
        for s in self.spikes:
            fn = s['frame']
            result[fn] = {
                "type": "spike",
                "enabled": True,
                "source": "auto",
                "magnitude": s.get('magnitude', 0),
                "local_median": s.get('local_median', 0),
                "ratio": s.get('ratio', 0),
            }

        # Auto-detected dips
        for d in self.dips:
            fn = d['frame']
            result[fn] = {
                "type": "dip",
                "enabled": True,
                "source": "auto",
                "magnitude": d.get('magnitude', 0),
                "local_median": d.get('local_median', 0),
                "ratio": d.get('ratio', 0),
            }

        # User overrides (highest priority)
        for fn, ov in self.user_overrides.items():
            if fn in result:
                result[fn]["type"] = ov["type"]
                result[fn]["enabled"] = ov["enabled"]
                result[fn]["source"] = "user"
            else:
                # User-added frame — look up motion data
                motion = 0.0
                local_avg = 0.0
                ratio = 0.0
                for mi, (frame_num, mag) in enumerate(self.magnitudes):
                    if frame_num == fn:
                        motion = mag
                        window = 7
                        start = max(0, mi - window)
                        end = min(len(self.magnitudes), mi + window + 1)
                        neighbors = [self.magnitudes[j][1] for j in range(start, end) if j != mi]
                        if neighbors:
                            local_avg = float(np.median(neighbors))
                            ratio = motion / local_avg if local_avg > 0 else 0
                        break
                result[fn] = {
                    "type": ov["type"],
                    "enabled": ov["enabled"],
                    "source": "user",
                    "magnitude": motion,
                    "local_median": local_avg,
                    "ratio": ratio,
                }

        return result

    def _get_enabled_spikes_dips(self):
        """Get sets of enabled spike and dip frame numbers, respecting ignore flags."""
        gt = self._get_ground_truth()
        ignore_spikes = self.ignore_spikes_var.get()
        ignore_dips = self.ignore_dips_var.get()
        spike_frames = set()
        dip_frames = set()
        for fn, info in gt.items():
            if not info["enabled"]:
                continue
            if info["type"] == "spike" and not ignore_spikes:
                spike_frames.add(fn)
            elif info["type"] == "dip" and not ignore_dips:
                dip_frames.add(fn)
        return spike_frames, dip_frames

    # ── Threshold handling ────────────────────────────────────────────

    def _on_threshold_drag(self, *args):
        """Fast path during slider drag — re-detect and redraw chart only."""
        if not self.magnitudes or self._in_draw:
            return
        spike_thresh = self.spike_threshold_var.get()
        dip_thresh = self.dip_threshold_var.get()
        self.spikes = self.detect_spikes(self.magnitudes, spike_thresh)
        self.dips = self.detect_dips(self.magnitudes, dip_thresh)
        self.draw_chart()

    def _on_slider_release(self, *args):
        """Full rebuild on slider release."""
        if not self.magnitudes:
            return
        self._on_threshold_drag()

    def _on_ignore_toggle(self, *args):
        """Redraw chart when ignore toggles change."""
        self.draw_chart()

    # ── Chart Drawing ─────────────────────────────────────────────────

    def draw_chart(self):
        """Draw motion magnitude chart with zoom support."""
        self._in_draw = True
        self.canvas.delete("all")
        self._cursor_line = None
        self._in_line = None
        self._out_line = None
        c = self.canvas
        cw = c.winfo_width()
        ch = c.winfo_height()
        if cw <= 1 or ch <= 1:
            c.update_idletasks()
            cw = c.winfo_width()
            ch = c.winfo_height()

        if not self.magnitudes or cw < 10 or ch < 10:
            self._in_draw = False
            return

        pad_left = 50
        pad_right = 20
        pad_top = 20
        pad_bottom = 30
        plot_w = cw - pad_left - pad_right
        plot_h = ch - pad_top - pad_bottom

        # Zoom range
        zs = self._zoom_start
        ze = self._zoom_end if self._zoom_end is not None else len(self.magnitudes)
        ze = min(ze, len(self.magnitudes))
        zs = max(0, zs)
        visible = list(range(zs, ze))
        n_vis = len(visible)
        if n_vis == 0:
            self._in_draw = False
            return

        self._chart_geom = {
            'pad_left': pad_left, 'pad_right': pad_right,
            'pad_top': pad_top, 'pad_bottom': pad_bottom,
            'plot_w': plot_w, 'plot_h': plot_h,
            'cw': cw, 'ch': ch,
            'n': n_vis, 'zoom_start': zs, 'zoom_end': ze
        }

        all_mags = [m[1] for m in self.magnitudes]
        vis_mags = [all_mags[i] for i in visible]
        max_mag = max(vis_mags) if vis_mags else 1

        # Build sets from ground truth
        gt = self._get_ground_truth()
        ignore_spikes = self.ignore_spikes_var.get()
        ignore_dips = self.ignore_dips_var.get()
        spike_frames = set()
        dip_frames = set()
        disabled_frames = set()
        for fn, info in gt.items():
            if not info["enabled"]:
                disabled_frames.add(fn)
            elif info["type"] == "spike" and ignore_spikes:
                continue
            elif info["type"] == "dip" and ignore_dips:
                continue
            elif info["type"] == "spike":
                spike_frames.add(fn)
            elif info["type"] == "dip":
                dip_frames.add(fn)

        # Axes
        c.create_line(pad_left, pad_top, pad_left, ch - pad_bottom, fill="#555")
        c.create_line(pad_left, ch - pad_bottom, cw - pad_right, ch - pad_bottom, fill="#555")

        # Y-axis labels
        c.create_text(pad_left - 5, pad_top, text=f"{max_mag:.1f}", anchor="ne",
                       fill="#888", font=("Consolas", 8))
        c.create_text(pad_left - 5, ch - pad_bottom, text="0", anchor="ne",
                       fill="#888", font=("Consolas", 8))

        # X-axis labels
        first_frame = self.magnitudes[zs][0]
        last_frame = self.magnitudes[ze - 1][0]
        c.create_text(pad_left, ch - pad_bottom + 12, text=str(first_frame), anchor="n",
                       fill="#888", font=("Consolas", 8))
        c.create_text(cw - pad_right, ch - pad_bottom + 12, text=str(last_frame), anchor="n",
                       fill="#888", font=("Consolas", 8))

        # Draw bars
        if n_vis > 0:
            bar_w = max(1, plot_w / n_vis)

            for vi, gi in enumerate(visible):
                mag = all_mags[gi]
                x = pad_left + (vi / n_vis) * plot_w
                h = (mag / max_mag) * plot_h if max_mag > 0 else 0
                y = ch - pad_bottom - h
                frame_num = self.magnitudes[gi][0]

                if frame_num in disabled_frames:
                    bar_color = "#555555"
                    label_color = "#777777"
                    c.create_rectangle(x, y, x + max(bar_w, 2), ch - pad_bottom,
                                        fill=bar_color, outline="")
                    if n_vis < 200:
                        c.create_text(x, y - 8, text=f"F{frame_num}", fill=label_color,
                                       font=("Consolas", 7), anchor="s")
                elif frame_num in spike_frames:
                    c.create_rectangle(x, y, x + max(bar_w, 2), ch - pad_bottom,
                                        fill="#ff4444", outline="")
                    if n_vis < 200:
                        c.create_text(x, y - 8, text=f"F{frame_num}", fill="#ff6b6b",
                                       font=("Consolas", 7), anchor="s")
                elif frame_num in dip_frames:
                    c.create_rectangle(x, y, x + max(bar_w, 2), ch - pad_bottom,
                                        fill="#ccaa00", outline="")
                    if n_vis < 200:
                        c.create_text(x, y - 8, text=f"F{frame_num}", fill="#ffcc44",
                                       font=("Consolas", 7), anchor="s")
                else:
                    c.create_rectangle(x, y, x + max(bar_w, 2), ch - pad_bottom,
                                        fill="#3a7acc", outline="")

                # Frame numbers on x-axis when zoomed
                if n_vis <= 60 and vi % max(1, n_vis // 15) == 0:
                    c.create_text(x, ch - pad_bottom + 12, text=str(frame_num),
                                   anchor="n", fill="#666", font=("Consolas", 7))

        self._draw_inout_markers()

        # Title
        zoom_info = f"  [{first_frame}\u2013{last_frame}]" if zs > 0 or ze < len(self.magnitudes) else ""
        c.create_text(cw // 2, 8, text=f"Motion Magnitude per Frame{zoom_info}  (click to lock)",
                       fill="#aaa", font=("Segoe UI", 9), anchor="n")

        self._draw_zoom_slider()
        self._in_draw = False

    # ── Chart Cursor ──────────────────────────────────────────────────

    def _chart_x_to_frame_index(self, x):
        g = getattr(self, '_chart_geom', None)
        if not g or not self.magnitudes:
            return None
        rel = x - g['pad_left']
        if rel < 0 or rel > g['plot_w']:
            return None
        vi = int((rel / g['plot_w']) * g['n'])
        vi = max(0, min(vi, g['n'] - 1))
        return g['zoom_start'] + vi

    def _global_idx_to_chart_x(self, idx):
        g = getattr(self, '_chart_geom', None)
        if not g:
            return None
        vi = idx - g['zoom_start']
        if vi < 0 or vi >= g['n']:
            return None
        return g['pad_left'] + (vi / g['n']) * g['plot_w']

    def _on_chart_hover(self, event):
        if self._cursor_locked:
            return
        self._update_cursor_from_x(event.x)

    def _on_chart_click(self, event):
        if self._cursor_locked:
            self._cursor_locked = False
            self._update_cursor_from_x(event.x)
        else:
            self._cursor_locked = True
            self._update_cursor_from_x(event.x)

    def _on_chart_right_click(self, event):
        """Right-click on chart bar for context menu."""
        idx = self._chart_x_to_frame_index(event.x)
        if idx is None or not self.magnitudes:
            return

        frame_num = self.magnitudes[idx][0]

        menu = tk.Menu(self.canvas, tearoff=0, bg="#333333", fg="#e0e0e0",
                       activebackground="#555555", activeforeground="#ffffff",
                       font=("Segoe UI", 9))

        gt = self._get_ground_truth()
        info = gt.get(frame_num)

        if info:
            # Frame is a spike or dip
            current_type = info["type"]
            is_enabled = info["enabled"]

            if current_type == "spike":
                menu.add_command(label="Spike \u2713", state=tk.DISABLED)
                menu.add_command(label="Set as Dip",
                                 command=lambda: self._ctx_set_type(frame_num, "dip"))
            else:
                menu.add_command(label="Set as Spike",
                                 command=lambda: self._ctx_set_type(frame_num, "spike"))
                menu.add_command(label="Dip \u2713", state=tk.DISABLED)

            menu.add_separator()

            if is_enabled:
                menu.add_command(label="Disable",
                                 command=lambda: self._ctx_set_enabled(frame_num, False))
            else:
                menu.add_command(label="Enable",
                                 command=lambda: self._ctx_set_enabled(frame_num, True))

            if frame_num in self.user_overrides:
                menu.add_separator()
                menu.add_command(label="Reset to Auto",
                                 command=lambda: self._ctx_reset(frame_num))
        else:
            # Normal frame — offer to mark
            menu.add_command(label="Mark as Spike",
                             command=lambda: self._ctx_add_frame(frame_num, "spike"))
            menu.add_command(label="Mark as Dip",
                             command=lambda: self._ctx_add_frame(frame_num, "dip"))

        menu.tk_popup(event.x_root, event.y_root)

    def _ctx_set_type(self, frame_num, new_type):
        ov = self.user_overrides.get(frame_num, {})
        enabled = ov.get("enabled", True)
        self.user_overrides[frame_num] = {"type": new_type, "enabled": enabled}
        self.draw_chart()

    def _ctx_set_enabled(self, frame_num, enabled):
        gt = self._get_ground_truth()
        info = gt.get(frame_num)
        frame_type = info["type"] if info else "spike"
        self.user_overrides[frame_num] = {"type": frame_type, "enabled": enabled}
        self.draw_chart()

    def _ctx_reset(self, frame_num):
        self.user_overrides.pop(frame_num, None)
        self.draw_chart()

    def _ctx_add_frame(self, frame_num, frame_type):
        self.user_overrides[frame_num] = {"type": frame_type, "enabled": True}
        self.draw_chart()

    # ── Chart Scroll (Zoom) ───────────────────────────────────────────

    def _on_chart_scroll(self, event):
        if not self.magnitudes:
            return
        g = getattr(self, '_chart_geom', None)
        if not g:
            return

        center_idx = self._chart_x_to_frame_index(event.x)
        if center_idx is None:
            center_idx = (g['zoom_start'] + g['zoom_end']) // 2

        total = len(self.magnitudes)
        current_span = g['zoom_end'] - g['zoom_start']

        if event.delta > 0:
            new_span = max(20, int(current_span * 0.7))
        else:
            new_span = min(total, int(current_span * 1.4))

        half = new_span // 2
        self._zoom_start = max(0, center_idx - half)
        self._zoom_end = min(total, self._zoom_start + new_span)
        if self._zoom_end >= total:
            self._zoom_start = max(0, total - new_span)
            self._zoom_end = total

        self.draw_chart()

    def _update_cursor_from_x(self, x):
        idx = self._chart_x_to_frame_index(x)
        if idx is not None:
            self._update_cursor(idx)

    def _update_cursor(self, idx):
        g = getattr(self, '_chart_geom', None)
        if not g or not self.magnitudes:
            return

        idx = max(0, min(idx, len(self.magnitudes) - 1))

        if idx == self._cursor_frame_idx:
            return
        self._cursor_frame_idx = idx

        c = self.canvas

        if self._cursor_line:
            c.delete(self._cursor_line)
            self._cursor_line = None

        snap_x = self._global_idx_to_chart_x(idx)
        if snap_x is not None:
            self._cursor_line = c.create_line(
                snap_x, g['pad_top'], snap_x, g['ch'] - g['pad_bottom'],
                fill="#ffdd44", width=2)

        frame_num = self.magnitudes[idx][0]
        mag_val = self.magnitudes[idx][1]
        gt = self._get_ground_truth()
        info = gt.get(frame_num)
        issue_tag = ""
        if info:
            if info["type"] == "spike":
                issue_tag = "  \u26a0 SPIKE"
            elif info["type"] == "dip":
                issue_tag = "  \u25bc DIP"
            if not info["enabled"]:
                issue_tag += " (disabled)"
        lock_tag = " \U0001f512" if self._cursor_locked else ""
        self.preview_label.configure(
            text=f"Frame {frame_num}  |  Motion: {mag_val:.2f}{issue_tag}{lock_tag}")
        self._show_frame_preview(idx)

    def _show_frame_preview(self, mag_idx):
        if not self._has_frames():
            return

        result = self._get_frame(mag_idx)
        if result is None:
            return

        _, bgr_frame = result
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        pc = self.preview_canvas
        pc.update_idletasks()
        pw = pc.winfo_width()
        ph = pc.winfo_height()
        if pw < 10 or ph < 10:
            pw, ph = 320, 320

        h, w = rgb.shape[:2]
        scale = min(pw / w, ph / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        from PIL import Image, ImageTk
        img = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(img)

        pc.delete("all")
        pc.create_image(pw // 2, ph // 2, image=photo, anchor="center")
        self._preview_photo = photo

    # ── Zoom Slider ───────────────────────────────────────────────────

    def _draw_zoom_slider(self):
        zs = self.zoom_slider
        zs.delete("all")
        zs.update_idletasks()
        w = zs.winfo_width()
        h = zs.winfo_height()
        if w < 20 or not self.magnitudes:
            return

        total = len(self.magnitudes)
        pad = 10
        track_w = w - pad * 2
        bar_y = h // 2
        handle_r = 7

        zs.create_rectangle(pad, bar_y - 3, w - pad, bar_y + 3,
                             fill="#333333", outline="")

        z_start = self._zoom_start
        z_end = self._zoom_end if self._zoom_end is not None else total
        left_x = pad + (z_start / total) * track_w
        right_x = pad + (z_end / total) * track_w

        zs.create_rectangle(left_x, bar_y - 5, right_x, bar_y + 5,
                             fill="#4a9eff", outline="", tags="bar")
        zs.create_oval(left_x - handle_r, bar_y - handle_r,
                        left_x + handle_r, bar_y + handle_r,
                        fill="#ffffff", outline="#888888", width=1, tags="left_handle")
        zs.create_oval(right_x - handle_r, bar_y - handle_r,
                        right_x + handle_r, bar_y + handle_r,
                        fill="#ffffff", outline="#888888", width=1, tags="right_handle")

        self._zoom_slider_geom = {
            'pad': pad, 'track_w': track_w, 'total': total,
            'left_x': left_x, 'right_x': right_x, 'handle_r': handle_r
        }

    def _zoom_slider_press(self, event):
        sg = getattr(self, '_zoom_slider_geom', None)
        if not sg:
            return

        x = event.x
        hr = sg['handle_r'] + 4

        if abs(x - sg['left_x']) <= hr:
            self._zoom_drag_mode = "left"
        elif abs(x - sg['right_x']) <= hr:
            self._zoom_drag_mode = "right"
        elif sg['left_x'] < x < sg['right_x']:
            self._zoom_drag_mode = "bar"
        else:
            self._zoom_drag_mode = "bar"
            total = sg['total']
            span = (self._zoom_end or total) - self._zoom_start
            clicked_idx = int(((x - sg['pad']) / sg['track_w']) * total)
            clicked_idx = max(0, min(total, clicked_idx))
            self._zoom_start = max(0, clicked_idx - span // 2)
            self._zoom_end = min(total, self._zoom_start + span)
            if self._zoom_end >= total:
                self._zoom_start = max(0, total - span)
            self.draw_chart()

        self._zoom_drag_start_x = x
        self._zoom_drag_start_vals = (self._zoom_start,
                                       self._zoom_end or len(self.magnitudes))

    def _zoom_slider_drag(self, event):
        sg = getattr(self, '_zoom_slider_geom', None)
        if not sg or not self._zoom_drag_mode:
            return

        total = sg['total']
        dx_pixels = event.x - self._zoom_drag_start_x
        dx_frames = int((dx_pixels / sg['track_w']) * total)
        orig_start, orig_end = self._zoom_drag_start_vals
        min_span = 10

        if self._zoom_drag_mode == "left":
            new_start = max(0, min(orig_start + dx_frames, orig_end - min_span))
            self._zoom_start = new_start
            self._zoom_end = orig_end
        elif self._zoom_drag_mode == "right":
            new_end = min(total, max(orig_end + dx_frames, orig_start + min_span))
            self._zoom_start = orig_start
            self._zoom_end = new_end
        elif self._zoom_drag_mode == "bar":
            span = orig_end - orig_start
            new_start = orig_start + dx_frames
            new_start = max(0, min(new_start, total - span))
            self._zoom_start = new_start
            self._zoom_end = new_start + span

        self.draw_chart()

    def _zoom_slider_release(self, event):
        self._zoom_drag_mode = None

    def _zoom_fit(self):
        self._zoom_start = 0
        self._zoom_end = None
        self.draw_chart()

    # ── Playback ──────────────────────────────────────────────────────

    def _toggle_play(self):
        if self._playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        if not self._has_frames() or not self.magnitudes:
            return
        self._playing = True
        self.play_btn.configure(text="\u23f8 Pause")
        self._cursor_locked = True

        if self._cursor_frame_idx is None:
            start = self._in_point if self._in_point is not None else 0
            self._cursor_frame_idx = start

        self._play_next_frame()

    def _stop_playback(self):
        self._playing = False
        self.play_btn.configure(text="\u25b6 Play")
        if self._play_after_id:
            self.root.after_cancel(self._play_after_id)
            self._play_after_id = None

    def _play_next_frame(self):
        if not self._playing or not self.magnitudes:
            return

        idx = self._cursor_frame_idx
        if idx is None:
            idx = 0

        start = self._in_point if self._in_point is not None else 0
        end = self._out_point if self._out_point is not None else len(self.magnitudes) - 1

        idx += 1
        if idx > end:
            if self.loop_var.get():
                idx = start
            else:
                self._stop_playback()
                return

        self._cursor_frame_idx = None
        self._update_cursor(idx)

        # Auto-scroll
        g = getattr(self, '_chart_geom', None)
        if g and (idx < g['zoom_start'] or idx >= g['zoom_end']):
            span = g['zoom_end'] - g['zoom_start']
            self._zoom_start = max(0, idx - span // 4)
            self._zoom_end = self._zoom_start + span
            self.draw_chart()

        speed_str = self.speed_var.get()
        speed = float(speed_str.replace('x', ''))
        if self.fps > 0:
            delay_ms = max(1, int(1000 / (self.fps * speed)))
        else:
            delay_ms = 42

        self._play_after_id = self.root.after(delay_ms, self._play_next_frame)

    def _step_frame(self, delta):
        if not self.magnitudes:
            return
        idx = self._cursor_frame_idx if self._cursor_frame_idx is not None else 0
        idx = max(0, min(len(self.magnitudes) - 1, idx + delta))
        self._cursor_locked = True
        self._cursor_frame_idx = None
        self._update_cursor(idx)

        g = getattr(self, '_chart_geom', None)
        if g and (idx < g['zoom_start'] or idx >= g['zoom_end']):
            span = g['zoom_end'] - g['zoom_start']
            self._zoom_start = max(0, idx - span // 4)
            self._zoom_end = self._zoom_start + span
            self.draw_chart()

    # ── In/Out Points ─────────────────────────────────────────────────

    def _set_in_point(self):
        if self._cursor_frame_idx is not None:
            self._in_point = self._cursor_frame_idx
            self._update_inout_label()
            self.draw_chart()

    def _set_out_point(self):
        if self._cursor_frame_idx is not None:
            self._out_point = self._cursor_frame_idx
            self._update_inout_label()
            self.draw_chart()

    def _clear_inout(self):
        self._in_point = None
        self._out_point = None
        self._update_inout_label()
        self.draw_chart()

    def _get_inout_frame_range(self):
        if self._in_point is None or self._out_point is None or not self.magnitudes:
            return None, None
        in_frame = self.magnitudes[min(self._in_point, self._out_point)][0]
        out_frame = self.magnitudes[max(self._in_point, self._out_point)][0]
        return in_frame, out_frame

    def _disable_inout_range(self):
        """Disable all spike/dip frames in the In-Out range."""
        in_f, out_f = self._get_inout_frame_range()
        if in_f is None:
            messagebox.showinfo("No range", "Set In and Out points on the chart first.")
            return
        gt = self._get_ground_truth()
        count = 0
        for fn, info in gt.items():
            if in_f <= fn <= out_f and info["enabled"]:
                self.user_overrides[fn] = {"type": info["type"], "enabled": False}
                count += 1
        self.log_msg(f"  Disabled {count} frames in range F{in_f}\u2013F{out_f}\n")
        self.draw_chart()

    def _enable_inout_range(self):
        """Enable all spike/dip frames in the In-Out range."""
        in_f, out_f = self._get_inout_frame_range()
        if in_f is None:
            messagebox.showinfo("No range", "Set In and Out points on the chart first.")
            return
        gt = self._get_ground_truth()
        count = 0
        for fn, info in gt.items():
            if in_f <= fn <= out_f and not info["enabled"]:
                self.user_overrides[fn] = {"type": info["type"], "enabled": True}
                count += 1
        self.log_msg(f"  Enabled {count} frames in range F{in_f}\u2013F{out_f}\n")
        self.draw_chart()

    def _update_inout_label(self):
        in_str = f"F{self.magnitudes[self._in_point][0]}" if self._in_point is not None and self.magnitudes else "\u2014"
        out_str = f"F{self.magnitudes[self._out_point][0]}" if self._out_point is not None and self.magnitudes else "\u2014"
        self.inout_label.configure(text=f"In: {in_str}  |  Out: {out_str}")

    def _draw_inout_markers(self):
        g = getattr(self, '_chart_geom', None)
        if not g:
            return
        c = self.canvas

        if self._in_point is not None:
            x = self._global_idx_to_chart_x(self._in_point)
            if x is not None:
                self._in_line = c.create_line(
                    x, g['pad_top'], x, g['ch'] - g['pad_bottom'],
                    fill="#44ff44", width=2, dash=(6, 3))
                c.create_text(x, g['pad_top'] - 2, text="IN", fill="#44ff44",
                               font=("Consolas", 7, "bold"), anchor="s")

        if self._out_point is not None:
            x = self._global_idx_to_chart_x(self._out_point)
            if x is not None:
                self._out_line = c.create_line(
                    x, g['pad_top'], x, g['ch'] - g['pad_bottom'],
                    fill="#ff4444", width=2, dash=(6, 3))
                c.create_text(x, g['pad_top'] - 2, text="OUT", fill="#ff4444",
                               font=("Consolas", 7, "bold"), anchor="s")

    # ── Burn Label ────────────────────────────────────────────────────

    def _burn_label(self, frame, text, position="bottom_left"):
        """Burn a text label into a frame (in-place)."""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.5, min(w, h) / 800)
        thickness = max(1, int(scale * 2))
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        margin = 8

        if position == "center":
            x = (w - tw) // 2
            y = (h + th) // 2
        else:
            x = margin
            y = h - margin

        cv2.rectangle(frame, (x - 4, y - th - 8), (x + tw + 4, y + baseline + 4),
                       (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness,
                     cv2.LINE_AA)

    def _burn_label_large(self, frame, text):
        """Burn large centered text on a frame."""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(1.5, min(w, h) / 400)
        thickness = max(2, int(scale * 3))
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        x = (w - tw) // 2
        y = (h + th) // 2
        cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness,
                     cv2.LINE_AA)

    # ── FFmpeg Writer ─────────────────────────────────────────────────

    class FFmpegWriter:
        def __init__(self, proc):
            self.proc = proc
            self._stderr_lines = []
            def _drain():
                for line in proc.stderr:
                    self._stderr_lines.append(line)
            self._drain_thread = threading.Thread(target=_drain, daemon=True)
            self._drain_thread.start()

        def write(self, frame):
            try:
                self.proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError) as e:
                stderr = b"".join(self._stderr_lines).decode(errors="replace")
                raise RuntimeError(f"FFmpeg pipe broke: {e}\n{stderr[-500:]}")

        def release(self):
            self.proc.stdin.close()
            self._drain_thread.join(timeout=10)
            self.proc.wait()
            if self.proc.returncode != 0:
                stderr = b"".join(self._stderr_lines).decode(errors="replace")
                raise RuntimeError(f"FFmpeg failed (code {self.proc.returncode}):\n{stderr[-500:]}")

    def _create_ffmpeg_video_writer(self, output_path, ffmpeg_path, use_prores):
        """Create an FFmpeg writer for video-only output."""
        if use_prores:
            cmd = [
                ffmpeg_path, "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{self.width}x{self.height}",
                "-r", str(self.fps), "-i", "-",
                "-c:v", "prores_ks", "-profile:v", "3",
                "-pix_fmt", "yuv422p10le", "-vendor", "apl0",
                "-an", output_path
            ]
        else:
            cmd = [
                ffmpeg_path, "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{self.width}x{self.height}",
                "-r", str(self.fps), "-i", "-",
                "-c:v", "libx264", "-preset", "slow",
                "-b:v", "25M", "-pix_fmt", "yuv420p",
                "-an", output_path
            ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                 creationflags=_SUBPROCESS_FLAGS)
        return self.FFmpegWriter(proc)

    def _create_cv2_writer(self, output_path):
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        return out

    # ── FFmpeg Chapters ───────────────────────────────────────────────

    def _write_chapters_file(self, chapter_list, path):
        """Write FFMETADATA1 chapters file.
        chapter_list: [(start_ms, end_ms, title), ...]
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(";FFMETADATA1\n")
            f.write("title=Frame Detective Analysis\n\n")
            for start_ms, end_ms, title in chapter_list:
                f.write("[CHAPTER]\n")
                f.write("TIMEBASE=1/1000\n")
                f.write(f"START={int(start_ms)}\n")
                f.write(f"END={int(end_ms)}\n")
                f.write(f"title={title}\n\n")

    def _mux_with_chapters_and_audio(self, video_path, output_path, chapters_path,
                                      audio_source=None, spike_times=None, ffmpeg_path=None):
        """Mux video with chapters and optionally audio (with silence insertion at spike points)."""
        if not ffmpeg_path:
            return

        if audio_source and spike_times:
            # Build audio filter to insert silence at spike insertion points
            filter_parts = self._build_audio_silence_filter(spike_times, audio_source, ffmpeg_path)
            if filter_parts:
                cmd = [
                    ffmpeg_path, "-y",
                    "-i", video_path,
                    "-i", audio_source,
                    "-i", chapters_path,
                    "-filter_complex", filter_parts,
                    "-map", "0:v", "-map", "[outa]", "-map_metadata", "2",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    output_path
                ]
            else:
                # Fallback: copy audio as-is
                cmd = [
                    ffmpeg_path, "-y",
                    "-i", video_path,
                    "-i", audio_source,
                    "-i", chapters_path,
                    "-map", "0:v", "-map", "1:a", "-map_metadata", "2",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", output_path
                ]
        elif audio_source:
            # No spikes, just copy audio
            cmd = [
                ffmpeg_path, "-y",
                "-i", video_path,
                "-i", audio_source,
                "-i", chapters_path,
                "-map", "0:v", "-map", "1:a", "-map_metadata", "2",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-shortest", output_path
            ]
        else:
            # No audio, just chapters
            cmd = [
                ffmpeg_path, "-y",
                "-i", video_path,
                "-i", chapters_path,
                "-map", "0:v", "-map_metadata", "1",
                "-c:v", "copy",
                output_path
            ]

        proc = subprocess.run(cmd, capture_output=True, creationflags=_SUBPROCESS_FLAGS)
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg mux failed:\n{proc.stderr.decode(errors='replace')[-500:]}")

    def _build_audio_silence_filter(self, spike_times_sec, audio_source, ffmpeg_path):
        """Build FFmpeg filter_complex string to insert silence at spike times.
        spike_times_sec: sorted list of times (seconds) where silence should be inserted.
        Returns filter string or None if too many spikes."""
        if not spike_times_sec or len(spike_times_sec) > 100:
            return None

        silence_dur = 1.0 / self.fps if self.fps > 0 else 1.0 / 24.0

        # Probe audio sample rate
        sample_rate = 48000
        try:
            probe = subprocess.run(
                [ffmpeg_path, "-i", audio_source, "-hide_banner"],
                capture_output=True, text=True, creationflags=_SUBPROCESS_FLAGS
            )
            for line in probe.stderr.split('\n'):
                if 'Audio:' in line:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p == 'Hz,' or p == 'Hz':
                            sample_rate = int(parts[i - 1])
                            break
                    break
        except Exception:
            pass

        # Build segments: audio trim + silence gap for each spike
        segments = []
        labels = []
        prev_time = 0.0

        for si, t in enumerate(spike_times_sec):
            # Audio segment before this spike
            seg_label = f"s{si}"
            segments.append(f"[1:a]atrim=start={prev_time:.6f}:end={t:.6f},asetpts=PTS-STARTPTS[{seg_label}]")
            labels.append(f"[{seg_label}]")

            # Silence segment
            gap_label = f"g{si}"
            segments.append(
                f"anullsrc=r={sample_rate}:cl=stereo,atrim=duration={silence_dur:.6f}[{gap_label}]"
            )
            labels.append(f"[{gap_label}]")
            prev_time = t

        # Final audio segment (from last spike to end)
        final_label = f"s{len(spike_times_sec)}"
        segments.append(f"[1:a]atrim=start={prev_time:.6f},asetpts=PTS-STARTPTS[{final_label}]")
        labels.append(f"[{final_label}]")

        n_segments = len(labels)
        concat_str = "".join(labels) + f"concat=n={n_segments}:v=0:a=1[outa]"
        return ";".join(segments) + ";" + concat_str

    def _has_audio(self, video_path, ffmpeg_path):
        """Check if a video file has an audio stream."""
        try:
            probe = subprocess.run(
                [ffmpeg_path, "-i", str(video_path), "-hide_banner"],
                capture_output=True, text=True, creationflags=_SUBPROCESS_FLAGS
            )
            return 'Audio:' in probe.stderr
        except Exception:
            return False

    # ── Export: Prep Video ────────────────────────────────────────────

    def export_prep(self):
        if not self.magnitudes or not self._has_frames():
            messagebox.showinfo("No data", "Analyze a video first.")
            return

        spike_frames, dip_frames = self._get_enabled_spikes_dips()
        if not spike_frames and not dip_frames:
            messagebox.showinfo("Nothing to do", "No enabled spikes or dips.")
            return

        out_fmt = self.output_format_var.get()
        use_prores = "ProRes" in out_fmt
        ext = ".mov" if use_prores else ".mp4"
        filetypes = [("QuickTime MOV", "*.mov")] if use_prores else [("MP4 files", "*.mp4")]

        output_path = filedialog.asksaveasfilename(
            title="Save Prep Video",
            defaultextension=ext,
            filetypes=filetypes,
            initialfile=self.video_path.stem + "_Prep" + ext
        )
        if not output_path:
            return

        self.set_status("Building Prep video...")
        self.prep_btn.configure(state=tk.DISABLED)

        def do_prep():
            try:
                ffmpeg_path = self._find_ffmpeg()
                has_ffmpeg = ffmpeg_path is not None
                has_audio = has_ffmpeg and self._has_audio(self.video_path, ffmpeg_path)

                # We'll write video to a temp file first, then mux with audio+chapters
                if has_ffmpeg:
                    temp_dir = tempfile.mkdtemp(prefix="fd_prep_")
                    temp_video = os.path.join(temp_dir, "video_only" + ext)
                    out = self._create_ffmpeg_video_writer(temp_video, ffmpeg_path, use_prores)
                else:
                    temp_dir = None
                    temp_video = None
                    out = self._create_cv2_writer(output_path)

                chapters = []
                spike_times_sec = []  # for audio silence insertion
                inserted = 0
                frame_counter = 0  # output frame counter (for chapters)

                # Build sorted list of spike/dip frame numbers for XML
                events = []  # [(output_frame_start, output_frame_end, original_frame, type), ...]

                total_to_process = self._frame_count()
                for i, (frame_idx, frame) in enumerate(self._iter_frames()):
                    # Insert black frame BEFORE spike frame
                    if frame_idx in spike_frames:
                        # Black frame with label
                        black = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                        self._burn_label_large(black, "SPIKE")
                        self._burn_label(black, f"F{frame_idx}")
                        out.write(black)

                        # Track for chapters and XML
                        time_ms = (frame_counter / self.fps) * 1000 if self.fps > 0 else 0
                        spike_times_sec.append(frame_counter / self.fps if self.fps > 0 else 0)
                        chapters.append((time_ms, time_ms + (1000 / self.fps if self.fps > 0 else 42),
                                         f"Spike - F{frame_idx}"))
                        events.append((frame_counter, frame_counter + 1, frame_idx, "spike"))
                        frame_counter += 1
                        inserted += 1

                    # Dip: write original frame but mark for chapters/XML
                    if frame_idx in dip_frames:
                        time_ms = (frame_counter / self.fps) * 1000 if self.fps > 0 else 0
                        chapters.append((time_ms, time_ms + (1000 / self.fps if self.fps > 0 else 42),
                                         f"Dip - F{frame_idx}"))
                        events.append((frame_counter, frame_counter + 1, frame_idx, "dip"))

                    # Write original frame
                    out.write(frame)
                    frame_counter += 1

                    pct = (i / total_to_process) * 100 if total_to_process > 0 else 100
                    self.root.after(0, lambda p=pct: self.progress_var.set(p))

                out.release()

                total_output_frames = frame_counter

                # Mux with chapters and audio
                if has_ffmpeg and temp_video:
                    chapters_file = os.path.join(temp_dir, "chapters.txt")
                    self._write_chapters_file(chapters, chapters_file)

                    audio_src = str(self.video_path) if has_audio else None
                    sorted_spike_times = sorted(spike_times_sec) if spike_times_sec else None

                    self.root.after(0, lambda: self.set_status("Muxing audio & chapters..."))
                    self._mux_with_chapters_and_audio(
                        temp_video, output_path, chapters_file,
                        audio_source=audio_src,
                        spike_times=sorted_spike_times,
                        ffmpeg_path=ffmpeg_path
                    )

                    # Cleanup temp
                    try:
                        os.remove(temp_video)
                        os.remove(chapters_file)
                        os.rmdir(temp_dir)
                    except Exception:
                        pass

                # Generate FCP7 XML
                xml_path = str(Path(output_path).with_suffix(".xml"))
                self._generate_fcp7_xml(
                    xml_path, output_path, total_output_frames,
                    events, is_prep=True
                )

                self.root.after(0, lambda: (
                    self.log_msg(f"\nPrep done! Inserted {inserted} black frames.\n", "good"),
                    self._log_clickable_path(output_path),
                    self.log_msg(f"XML: ", "good"),
                    self._log_clickable_path(xml_path),
                    self.set_status(f"Saved: {output_path}"),
                    self.progress_var.set(100),
                    self.prep_btn.configure(state=tk.NORMAL)
                ))

            except Exception as e:
                import traceback
                err_msg = f"{str(e)}\n\n{traceback.format_exc()}"
                self.root.after(0, lambda: (
                    messagebox.showerror("Error", err_msg),
                    self.log_msg(f"\nERROR: {err_msg}\n", "error"),
                    self.prep_btn.configure(state=tk.NORMAL)
                ))

        threading.Thread(target=do_prep, daemon=True).start()

    # ── Export: Map Video ─────────────────────────────────────────────

    def export_map(self):
        if not self.magnitudes or not self._has_frames():
            messagebox.showinfo("No data", "Analyze a video first.")
            return

        out_fmt = self.output_format_var.get()
        use_prores = "ProRes" in out_fmt
        ext = ".mov" if use_prores else ".mp4"
        filetypes = [("QuickTime MOV", "*.mov")] if use_prores else [("MP4 files", "*.mp4")]

        output_path = filedialog.asksaveasfilename(
            title="Save Map Video",
            defaultextension=ext,
            filetypes=filetypes,
            initialfile=self.video_path.stem + "_Map" + ext
        )
        if not output_path:
            return

        self.set_status("Building Map video...")
        self.map_btn.configure(state=tk.DISABLED)

        def do_map():
            try:
                ffmpeg_path = self._find_ffmpeg()
                has_ffmpeg = ffmpeg_path is not None

                if has_ffmpeg:
                    temp_dir = tempfile.mkdtemp(prefix="fd_map_")
                    temp_video = os.path.join(temp_dir, "video_only" + ext)
                    out = self._create_ffmpeg_video_writer(temp_video, ffmpeg_path, use_prores)
                else:
                    temp_dir = None
                    temp_video = None
                    out = self._create_cv2_writer(output_path)

                spike_frames, dip_frames = self._get_enabled_spikes_dips()
                chapters = []
                events = []  # [(frame_start, frame_end, original_frame, type), ...]

                # Build magnitude lookup
                mag_lookup = {}
                for frame_idx, mag in self.magnitudes:
                    mag_lookup[frame_idx] = mag

                total_to_process = self._frame_count()
                for i, (frame_idx, _) in enumerate(self._iter_frames()):
                    black = np.zeros((self.height, self.width, 3), dtype=np.uint8)

                    # Velocity in bottom-left
                    vel = mag_lookup.get(frame_idx, 0.0)
                    self._burn_label(black, f"vel: {vel:.2f}  F{frame_idx}")

                    if frame_idx in spike_frames:
                        self._burn_label_large(black, "SPIKE")
                        time_ms = (i / self.fps) * 1000 if self.fps > 0 else 0
                        chapters.append((time_ms, time_ms + (1000 / self.fps if self.fps > 0 else 42),
                                         f"Spike - F{frame_idx}"))
                        events.append((i, i + 1, frame_idx, "spike"))
                    elif frame_idx in dip_frames:
                        self._burn_label_large(black, "DIP")
                        time_ms = (i / self.fps) * 1000 if self.fps > 0 else 0
                        chapters.append((time_ms, time_ms + (1000 / self.fps if self.fps > 0 else 42),
                                         f"Dip - F{frame_idx}"))
                        events.append((i, i + 1, frame_idx, "dip"))

                    out.write(black)

                    pct = (i / total_to_process) * 100 if total_to_process > 0 else 100
                    self.root.after(0, lambda p=pct: self.progress_var.set(p))

                out.release()

                total_output_frames = self._frame_count()

                # Mux with chapters (no audio for map)
                if has_ffmpeg and temp_video:
                    chapters_file = os.path.join(temp_dir, "chapters.txt")
                    self._write_chapters_file(chapters, chapters_file)

                    self.root.after(0, lambda: self.set_status("Adding chapters..."))
                    self._mux_with_chapters_and_audio(
                        temp_video, output_path, chapters_file,
                        ffmpeg_path=ffmpeg_path
                    )

                    try:
                        os.remove(temp_video)
                        os.remove(chapters_file)
                        os.rmdir(temp_dir)
                    except Exception:
                        pass

                # Generate FCP7 XML
                xml_path = str(Path(output_path).with_suffix(".xml"))
                self._generate_fcp7_xml(
                    xml_path, output_path, total_output_frames,
                    events, is_prep=False
                )

                n_spikes = sum(1 for e in events if e[3] == "spike")
                n_dips = sum(1 for e in events if e[3] == "dip")

                self.root.after(0, lambda: (
                    self.log_msg(f"\nMap done! {n_spikes} spikes, {n_dips} dips marked.\n", "good"),
                    self._log_clickable_path(output_path),
                    self.log_msg(f"XML: ", "good"),
                    self._log_clickable_path(xml_path),
                    self.set_status(f"Saved: {output_path}"),
                    self.progress_var.set(100),
                    self.map_btn.configure(state=tk.NORMAL)
                ))

            except Exception as e:
                import traceback
                err_msg = f"{str(e)}\n\n{traceback.format_exc()}"
                self.root.after(0, lambda: (
                    messagebox.showerror("Error", err_msg),
                    self.log_msg(f"\nERROR: {err_msg}\n", "error"),
                    self.map_btn.configure(state=tk.NORMAL)
                ))

        threading.Thread(target=do_map, daemon=True).start()

    # ── Export: XML Only ──────────────────────────────────────────────

    def export_xml_only(self):
        if not self.magnitudes:
            messagebox.showinfo("No data", "Analyze a video first.")
            return

        spike_frames, dip_frames = self._get_enabled_spikes_dips()
        if not spike_frames and not dip_frames:
            messagebox.showinfo("Nothing to do", "No enabled spikes or dips.")
            return

        # Ask which type
        xml_type = tk.StringVar(value="prep")
        dlg = tk.Toplevel(self.root)
        dlg.title("Export XML")
        dlg.geometry("300x150")
        dlg.configure(bg="#1a1a1a")
        dlg.transient(self.root)
        dlg.grab_set()

        ttk.Label(dlg, text="XML type:", style="Dark.TLabel").pack(pady=(16, 4))
        ttk.Radiobutton(dlg, text="Prep (with inserted frames)", variable=xml_type,
                         value="prep").pack(anchor=tk.W, padx=30)
        ttk.Radiobutton(dlg, text="Map (original frame count)", variable=xml_type,
                         value="map").pack(anchor=tk.W, padx=30)

        def do_export():
            dlg.destroy()
            is_prep = xml_type.get() == "prep"

            out_fmt = self.output_format_var.get()
            ext = ".mov" if "ProRes" in out_fmt else ".mp4"
            suffix = "_Prep" if is_prep else "_Map"
            video_ref = self.video_path.stem + suffix + ext

            xml_path = filedialog.asksaveasfilename(
                title="Save XML",
                defaultextension=".xml",
                filetypes=[("FCP7 XML", "*.xml")],
                initialfile=self.video_path.stem + suffix + ".xml"
            )
            if not xml_path:
                return

            # Build events list (uses magnitudes for frame indices, no pixel data needed)
            events = []
            if is_prep:
                frame_counter = 0
                for frame_idx, _ in self.magnitudes:
                    if frame_idx in spike_frames:
                        events.append((frame_counter, frame_counter + 1, frame_idx, "spike"))
                        frame_counter += 1
                    if frame_idx in dip_frames:
                        events.append((frame_counter, frame_counter + 1, frame_idx, "dip"))
                    frame_counter += 1
                total_frames = frame_counter
            else:
                for i, (frame_idx, _) in enumerate(self.magnitudes):
                    if frame_idx in spike_frames:
                        events.append((i, i + 1, frame_idx, "spike"))
                    elif frame_idx in dip_frames:
                        events.append((i, i + 1, frame_idx, "dip"))
                total_frames = len(self.magnitudes)

            # Reference the video file (may not exist)
            video_path_ref = str(Path(xml_path).parent / video_ref)

            self._generate_fcp7_xml(xml_path, video_path_ref, total_frames, events, is_prep=is_prep)
            self.log_msg(f"\nXML exported: ", "good")
            self._log_clickable_path(xml_path)

        ttk.Button(dlg, text="Export", command=do_export, style="Accent.TButton").pack(pady=12)

    # ── FCP7 XML Generation ───────────────────────────────────────────

    def _generate_fcp7_xml(self, xml_path, video_path, total_frames, events, is_prep=True):
        """Generate FCP7 XML (xmeml v5) for Premiere Pro / DaVinci Resolve.

        events: [(output_frame_start, output_frame_end, original_frame_num, "spike"|"dip"), ...]
        """
        # Determine rate
        ntsc, timebase = self._get_fcp_rate()
        duration = total_frames

        video_name = Path(video_path).stem
        video_filename = Path(video_path).name

        # Convert to file URL (percent-encode spaces and special chars)
        abs_path = os.path.abspath(video_path)
        if sys.platform == "win32":
            path_parts = abs_path.replace("\\", "/")
            file_url = "file://localhost/" + url_quote(path_parts, safe="/:@")
        else:
            file_url = "file://localhost" + url_quote(abs_path, safe="/:@")

        # Build xmeml
        root = ET.Element("xmeml", version="5")
        seq = ET.SubElement(root, "sequence")
        ET.SubElement(seq, "name").text = f"Frame Detective - {video_name}"
        ET.SubElement(seq, "duration").text = str(duration)

        rate_el = ET.SubElement(seq, "rate")
        ET.SubElement(rate_el, "ntsc").text = "TRUE" if ntsc else "FALSE"
        ET.SubElement(rate_el, "timebase").text = str(timebase)

        # Sequence-level markers (show as timeline markers in Premiere Pro)
        for evt_start, evt_end, orig_frame, evt_type in sorted(events, key=lambda e: e[0]):
            marker = ET.SubElement(seq, "marker")
            if evt_type == "spike":
                ET.SubElement(marker, "name").text = f"Spike - F{orig_frame}"
                ET.SubElement(marker, "comment").text = "spike"
                ET.SubElement(marker, "color").text = "Red"
            else:
                ET.SubElement(marker, "name").text = f"Dip - F{orig_frame}"
                ET.SubElement(marker, "comment").text = "dip"
                ET.SubElement(marker, "color").text = "Yellow"
            ET.SubElement(marker, "in").text = str(evt_start)
            ET.SubElement(marker, "out").text = str(-1)

        media = ET.SubElement(seq, "media")
        video_el = ET.SubElement(media, "video")
        track = ET.SubElement(video_el, "track")

        # Build clip list: normal segments + spike/dip events
        # Sort events by output frame start
        sorted_events = sorted(events, key=lambda e: e[0])

        clips = []
        prev_end = 0

        for evt_start, evt_end, orig_frame, evt_type in sorted_events:
            # Normal segment before this event
            if evt_start > prev_end:
                clips.append({
                    "name": "Normal",
                    "in": prev_end,
                    "out": evt_start,
                    "start": prev_end,
                    "end": evt_start,
                })

            # Event clip
            label = f"Spike - F{orig_frame}" if evt_type == "spike" else f"Dip - F{orig_frame}"
            clips.append({
                "name": label,
                "in": evt_start,
                "out": evt_end,
                "start": evt_start,
                "end": evt_end,
            })
            prev_end = evt_end

        # Final normal segment
        if prev_end < duration:
            clips.append({
                "name": "Normal",
                "in": prev_end,
                "out": duration,
                "start": prev_end,
                "end": duration,
            })

        # Write clips to XML
        file_defined = False
        for ci, clip in enumerate(clips):
            clipitem = ET.SubElement(track, "clipitem", id=f"clip{ci + 1}")
            ET.SubElement(clipitem, "name").text = clip["name"]
            ET.SubElement(clipitem, "duration").text = str(duration)

            clip_rate = ET.SubElement(clipitem, "rate")
            ET.SubElement(clip_rate, "ntsc").text = "TRUE" if ntsc else "FALSE"
            ET.SubElement(clip_rate, "timebase").text = str(timebase)

            ET.SubElement(clipitem, "in").text = str(clip["in"])
            ET.SubElement(clipitem, "out").text = str(clip["out"])
            ET.SubElement(clipitem, "start").text = str(clip["start"])
            ET.SubElement(clipitem, "end").text = str(clip["end"])

            if not file_defined:
                # First clip: define the file element with full metadata
                file_el = ET.SubElement(clipitem, "file", id="file1")
                ET.SubElement(file_el, "name").text = video_filename
                ET.SubElement(file_el, "pathurl").text = file_url

                file_rate = ET.SubElement(file_el, "rate")
                ET.SubElement(file_rate, "ntsc").text = "TRUE" if ntsc else "FALSE"
                ET.SubElement(file_rate, "timebase").text = str(timebase)

                ET.SubElement(file_el, "duration").text = str(duration)

                file_media = ET.SubElement(file_el, "media")
                file_video = ET.SubElement(file_media, "video")
                sample = ET.SubElement(file_video, "samplecharacteristics")
                ET.SubElement(sample, "width").text = str(self.width)
                ET.SubElement(sample, "height").text = str(self.height)

                file_audio = ET.SubElement(file_media, "audio")
                audio_sample = ET.SubElement(file_audio, "samplecharacteristics")
                ET.SubElement(audio_sample, "samplerate").text = "48000"
                ET.SubElement(audio_sample, "depth").text = "16"

                file_defined = True
            else:
                # Subsequent clips: reference existing file
                ET.SubElement(clipitem, "file", id="file1")

        # Pretty-print
        rough_string = ET.tostring(root, encoding="unicode")
        # Wrap with XML declaration and DOCTYPE
        xml_string = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n'
        try:
            dom = minidom.parseString(rough_string)
            pretty = dom.toprettyxml(indent="  ")
            # Remove minidom's xml declaration (we add our own)
            lines = pretty.split('\n')
            if lines[0].startswith('<?xml'):
                lines = lines[1:]
            xml_string += '\n'.join(lines)
        except Exception:
            xml_string += rough_string

        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(xml_string)

    def _get_fcp_rate(self):
        """Determine FCP ntsc flag and timebase from source fps."""
        fps = self.fps
        if abs(fps - 29.97) < 0.1 or abs(fps - 59.94) < 0.1:
            ntsc = True
            timebase = round(fps * 1000 / 999)  # 30 or 60
        elif abs(fps - 23.976) < 0.1:
            ntsc = True
            timebase = 24
        else:
            ntsc = False
            timebase = round(fps)
        return ntsc, timebase

    # ── Project Save/Load ─────────────────────────────────────────────

    def _build_project_data(self):
        return {
            "version": 4,
            "app": "Frame Detective V3",
            "video_path": str(self.video_path) if self.video_path else None,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames,
            "magnitudes": self.magnitudes,
            "spikes": self.spikes,
            "dips": self.dips,
            "spike_threshold": self.spike_threshold_var.get(),
            "dip_threshold": self.dip_threshold_var.get(),
            "ignore_spikes": self.ignore_spikes_var.get(),
            "ignore_dips": self.ignore_dips_var.get(),
            "output_format": self.output_format_var.get(),
            "user_overrides": {str(f): v for f, v in self.user_overrides.items()},
            "in_point": self._in_point,
            "out_point": self._out_point,
        }

    def save_project(self):
        if not self.magnitudes:
            messagebox.showinfo("Nothing to save", "Analyze a video first.")
            return
        if self._project_path:
            self._write_project(self._project_path)
        else:
            self.save_project_as()

    def save_project_as(self):
        if not self.magnitudes:
            messagebox.showinfo("Nothing to save", "Analyze a video first.")
            return
        initial = self.video_path.stem + ".fdp" if self.video_path else "project.fdp"
        path = filedialog.asksaveasfilename(
            title="Save Project",
            defaultextension=".fdp",
            filetypes=[("Frame Detective Project", "*.fdp")],
            initialfile=initial
        )
        if not path:
            return
        self._project_path = path
        self._write_project(path)

    def _write_project(self, path):
        data = self._build_project_data()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self.log_msg(f"Project saved: {Path(path).name}\n", "good")
        self.set_status(f"Saved: {path}")
        self.root.title(f"Frame Detective V3 \u2014 {Path(path).name}")

    def open_project(self):
        path = filedialog.askopenfilename(
            title="Open Project",
            filetypes=[("Frame Detective Project", "*.fdp"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Could not read project file:\n{e}")
            return

        version = data.get("version", 0)
        if version not in (3, 4):
            messagebox.showwarning("Version", "This project file may be from a different version.")

        self._project_path = path

        # Reset state
        self._in_point = data.get("in_point")
        self._out_point = data.get("out_point")
        self._cursor_frame_idx = None
        self._cursor_locked = False
        self._zoom_start = 0
        self._zoom_end = None

        # Restore video path
        vp = data.get("video_path")
        if vp:
            self.video_path = Path(vp)
            self.file_label.configure(text=self.video_path.name)

        # Restore analysis data
        self.fps = data.get("fps", 0)
        self.width = data.get("width", 0)
        self.height = data.get("height", 0)
        self.total_frames = data.get("total_frames", 0)
        self.magnitudes = [tuple(m) for m in data.get("magnitudes", [])]
        self.spikes = data.get("spikes", [])
        self.dips = data.get("dips", [])

        # Restore settings
        self.spike_threshold_var.set(data.get("spike_threshold", 2.0))
        self.dip_threshold_var.set(data.get("dip_threshold", 0.70))
        self.ignore_spikes_var.set(data.get("ignore_spikes", False))
        self.ignore_dips_var.set(data.get("ignore_dips", False))
        self.output_format_var.set(data.get("output_format", "H.264 (.mp4)"))

        # Restore user overrides (convert v3 format if needed)
        raw_overrides = data.get("user_overrides", {})
        self.user_overrides = {}
        for f_str, v in raw_overrides.items():
            fn = int(f_str)
            if isinstance(v, dict) and "enabled" in v:
                self.user_overrides[fn] = v
            elif isinstance(v, dict) and "count" in v:
                # V3 compat: convert count-based to enabled-based
                self.user_overrides[fn] = {
                    "type": v.get("type", "spike"),
                    "enabled": v.get("count", 1) > 0
                }

        # Reload frame data from video
        self._cleanup_disk_cache()
        self.frames_data = []
        if self.disk_cache_var.get():
            self._init_disk_cache()
        if self.video_path and self.video_path.exists():
            cap = cv2.VideoCapture(str(self.video_path))
            if cap.isOpened():
                idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    self._store_frame(idx, frame)
                    idx += 1
                cap.release()
                self.log_msg(f"Reloaded {self._frame_count()} frames from video.\n")
            else:
                self.log_msg(f"Warning: could not open video {self.video_path}\n", "error")
        else:
            self.log_msg(f"Warning: video file not found \u2014 preview/output unavailable.\n", "error")

        self.draw_chart()
        self._update_inout_label()

        self.log_msg(f"Opened project: {Path(path).name}\n", "good")
        self.set_status(f"Opened: {path}")
        self.root.title(f"Frame Detective V3 \u2014 {Path(path).name}")


def main():
    root = tk.Tk()
    app = FrameDetectiveApp(root)

    def on_close():
        app._cleanup_disk_cache()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
