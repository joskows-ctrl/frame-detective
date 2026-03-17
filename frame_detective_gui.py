"""
Frame Detective GUI — Detect & fix missing frames in AI-generated video
Drag or browse for an MP4, analyze motion spikes, insert black frames to smooth.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading
import subprocess
import shutil
import webbrowser
import sys


class FrameDetectiveApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Frame Detective V1")
        self.root.geometry("1100x920")
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

        self.video_path = None
        self.magnitudes = []
        self.frames_data = []
        self.spikes = []
        self.dips = []
        self.dip_entries = set()
        self.fps = 0
        self.width = 0
        self.height = 0
        self.analyzing = False

        self.setup_styles()
        self.build_ui()

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
        style.configure("Spike.TLabel", background="#1a1a1a", foreground="#ff6b6b",
                         font=("Segoe UI", 10, "bold"))
        style.configure("Dark.TButton", font=("Segoe UI", 10))
        style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"))
        style.configure("Dark.TLabelframe", background="#1a1a1a", foreground="#e0e0e0")
        style.configure("Dark.TLabelframe.Label", background="#1a1a1a", foreground="#e0e0e0",
                         font=("Segoe UI", 10, "bold"))

    def build_ui(self):
        # Main container
        main = ttk.Frame(self.root, style="Dark.TFrame", padding=20)
        main.pack(fill=tk.BOTH, expand=True)

        # Title row with logo
        title_row = ttk.Frame(main, style="Dark.TFrame")
        title_row.pack(fill=tk.X, pady=(0, 8))

        # Load logo image
        self._logo_img = None
        try:
            from PIL import Image, ImageTk
            if getattr(sys, 'frozen', False):
                logo_path = Path(sys._MEIPASS) / "FDIcon.png"
            else:
                logo_path = Path(__file__).parent / "FDIcon.png"
            if logo_path.exists():
                logo_raw = Image.open(str(logo_path))
                # Scale to 160px tall, keep aspect ratio
                target_h = 160
                ratio = logo_raw.size[0] / logo_raw.size[1]
                target_w = int(target_h * ratio)
                logo_resized = logo_raw.resize((target_w, target_h), Image.LANCZOS)
                self._logo_img = ImageTk.PhotoImage(logo_resized)
        except Exception:
            pass

        if self._logo_img:
            logo_label = tk.Label(title_row, image=self._logo_img, bg="#1a1a1a")
            logo_label.pack(side=tk.LEFT)
        else:
            ttk.Label(title_row, text="Frame Detective V1", style="Title.TLabel").pack(side=tk.LEFT)

        # Website link button
        ha_btn = tk.Button(title_row, text="highlyappropriate.com", fg="#6699cc",
                            bg="#1a1a1a", activeforeground="#88bbee", activebackground="#1a1a1a",
                            bd=0, cursor="hand2", font=("Segoe UI", 9, "underline"),
                            command=lambda: webbrowser.open("https://highlyappropriate.com"))
        ha_btn.pack(side=tk.RIGHT, anchor="s", pady=(0, 4))

        # File selection row
        file_frame = ttk.Frame(main, style="Dark.TFrame")
        file_frame.pack(fill=tk.X, pady=(0, 12))

        self.file_label = ttk.Label(file_frame, text="No file selected",
                                     style="Dark.TLabel", width=60)
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        browse_btn = ttk.Button(file_frame, text="Browse Video...", command=self.browse_file)
        browse_btn.pack(side=tk.RIGHT, padx=(8, 0))

        # Settings row
        settings_frame = ttk.LabelFrame(main, text="Settings", style="Dark.TLabelframe",
                                         padding=12)
        settings_frame.pack(fill=tk.X, pady=(0, 12))

        settings_inner = ttk.Frame(settings_frame, style="Dark.TFrame")
        settings_inner.pack(fill=tk.X)

        # Detection Threshold
        thresh_label = ttk.Label(settings_inner, text="Detection Threshold:", style="Dark.TLabel")
        thresh_label.pack(side=tk.LEFT)
        self._add_tooltip(thresh_label, "How much motion a frame needs vs its neighbors to be flagged.\n"
                                         "2.0 = frame must have 2x the normal motion.\n"
                                         "Lower = more sensitive, higher = only big jumps.")
        self.threshold_var = tk.DoubleVar(value=2.0)
        thresh_spin = ttk.Spinbox(settings_inner, from_=1.2, to=5.0, increment=0.1,
                                   textvariable=self.threshold_var, width=6)
        thresh_spin.pack(side=tk.LEFT, padx=(4, 4))
        thresh_hint = ttk.Label(settings_inner, text="?", style="Dark.TLabel",
                                 font=("Segoe UI", 9, "bold"), cursor="question_arrow")
        thresh_hint.pack(side=tk.LEFT, padx=(0, 16))
        self._add_tooltip(thresh_hint, "How much motion a frame needs vs its neighbors to be flagged.\n"
                                        "2.0 = frame must have 2x the normal motion.\n"
                                        "Lower = more sensitive, higher = only big jumps.")

        # Second settings row — fill mode + precision
        settings_inner2 = ttk.Frame(settings_frame, style="Dark.TFrame")
        settings_inner2.pack(fill=tk.X, pady=(8, 0))

        ttk.Label(settings_inner2, text="Fill mode:", style="Dark.TLabel").pack(side=tk.LEFT)
        self.fill_mode_var = tk.StringVar(value="RIFE (AI)")
        self.fill_mode_combo = ttk.Combobox(settings_inner2, textvariable=self.fill_mode_var,
                                             values=["Black", "White", "Blend (debug)", "RIFE (AI)", "RIFE + Labels"],
                                             state="readonly", width=18)
        self.fill_mode_combo.pack(side=tk.LEFT, padx=(4, 0))

        # Full precision toggle (for RIFE)
        self.full_precision_var = tk.BooleanVar(value=True)
        self.fp_cb = ttk.Checkbutton(settings_inner2, text="Full precision",
                                      variable=self.full_precision_var)
        self.fp_cb.pack(side=tk.LEFT, padx=(12, 0))

        # Third settings row — fix mode
        settings_inner_fix = ttk.Frame(settings_frame, style="Dark.TFrame")
        settings_inner_fix.pack(fill=tk.X, pady=(8, 0))

        ttk.Label(settings_inner_fix, text="Fix mode:", style="Dark.TLabel").pack(side=tk.LEFT)
        self.fix_mode_var = tk.StringVar(value="Spikes + Dips")
        self.fix_mode_combo = ttk.Combobox(settings_inner_fix, textvariable=self.fix_mode_var,
                                            values=["Spikes Only", "Dips Only", "Spikes + Dips"],
                                            state="readonly", width=14)
        self.fix_mode_combo.pack(side=tk.LEFT, padx=(4, 0))
        fix_mode_hint = ttk.Label(settings_inner_fix, text="?", style="Dark.TLabel",
                                   font=("Segoe UI", 9, "bold"), cursor="question_arrow")
        fix_mode_hint.pack(side=tk.LEFT, padx=(4, 0))
        self._add_tooltip(fix_mode_hint, "Spikes = missing frames (insert interpolated frames)\n"
                                          "Dips = duplicate/stuck frames (replace with interpolation)\n"
                                          "Spikes + Dips = fix both in one pass")

        # Fourth settings row — output format
        settings_inner3 = ttk.Frame(settings_frame, style="Dark.TFrame")
        settings_inner3.pack(fill=tk.X, pady=(8, 0))

        ttk.Label(settings_inner3, text="Output:", style="Dark.TLabel").pack(side=tk.LEFT)
        self.output_format_var = tk.StringVar(value="H.264 (.mp4)")
        formats = ["H.264 (.mp4)", "ProRes HQ (.mov)"]
        if not shutil.which("ffmpeg"):
            formats = ["H.264 (.mp4)"]  # ProRes needs FFmpeg
        self.output_format_combo = ttk.Combobox(settings_inner3, textvariable=self.output_format_var,
                                                 values=formats, state="readonly", width=18)
        self.output_format_combo.pack(side=tk.LEFT, padx=(4, 0))

        self.ffmpeg_label = ttk.Label(settings_inner3,
                                       text="✓ FFmpeg found" if shutil.which("ffmpeg") else "⚠ FFmpeg not found (ProRes unavailable)",
                                       style="Info.TLabel")
        self.ffmpeg_label.pack(side=tk.LEFT, padx=(12, 0))

        # Buttons row 1: Analyze + Fix
        btn_frame = ttk.Frame(main, style="Dark.TFrame")
        btn_frame.pack(fill=tk.X, pady=(0, 4))

        self.analyze_btn = ttk.Button(btn_frame, text="Analyze", command=self.start_analyze,
                                       style="Accent.TButton")
        self.analyze_btn.pack(side=tk.LEFT)

        self.fix_btn = ttk.Button(btn_frame, text="Fix Video (Insert Frames)",
                                   command=self.fix_video, state=tk.DISABLED)
        self.fix_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.export_btn = ttk.Button(btn_frame, text="Export Report",
                                      command=self.export_report, state=tk.DISABLED)
        self.export_btn.pack(side=tk.LEFT, padx=(8, 0))

        # Buttons row 2: Recursive Fix + Intensity + Stop (hidden for now)
        btn_frame2 = ttk.Frame(main, style="Dark.TFrame")
        # btn_frame2.pack(fill=tk.X, pady=(0, 8))  # Hidden — recursive mode shelved

        self.recursive_btn = ttk.Button(btn_frame2, text="Recursive Fix",
                                        command=self.recursive_fix, state=tk.DISABLED)
        self.recursive_btn.pack(side=tk.LEFT)

        self.intensity_var = tk.IntVar(value=5)
        self.stop_btn = ttk.Button(btn_frame2, text="Stop", command=self.stop_recursive,
                                    state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(12, 0))

        # Progress
        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(main, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, pady=(0, 4))

        self.status_label = ttk.Label(main, text="Ready", style="Info.TLabel")
        self.status_label.pack(anchor="w", pady=(0, 8))

        # Results area — chart + spike table + log on left, frame preview on right
        results_frame = ttk.LabelFrame(main, text="Results", style="Dark.TLabelframe",
                                        padding=8)
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Split: left side (chart, table, log) and right side (frame preview)
        results_split = ttk.Frame(results_frame, style="Dark.TFrame")
        results_split.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(results_split, style="Dark.TFrame")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_panel = ttk.Frame(results_split, style="Dark.TFrame")
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))

        # --- Right panel: Frame preview + playback ---
        ttk.Label(right_panel, text="Frame Preview", style="Dark.TLabel",
                   font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.preview_label = ttk.Label(right_panel, text="Hover chart to preview", style="Info.TLabel")
        self.preview_label.pack(anchor="w", pady=(2, 4))
        self.preview_canvas = tk.Canvas(right_panel, bg="#111111", width=320, height=320,
                                         highlightthickness=1, highlightbackground="#333333")
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, pady=(0, 4))
        self._preview_photo = None  # hold reference to prevent GC

        # Playback controls
        playback_frame = ttk.Frame(right_panel, style="Dark.TFrame")
        playback_frame.pack(fill=tk.X, pady=(0, 4))

        self.play_btn = ttk.Button(playback_frame, text="▶ Play", command=self._toggle_play, width=8)
        self.play_btn.pack(side=tk.LEFT)

        self.speed_var = tk.StringVar(value="1x")
        speed_combo = ttk.Combobox(playback_frame, textvariable=self.speed_var,
                                    values=["0.25x", "0.5x", "1x", "2x"], state="readonly", width=5)
        speed_combo.pack(side=tk.LEFT, padx=(4, 0))

        self.loop_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(playback_frame, text="Loop", variable=self.loop_var).pack(side=tk.LEFT, padx=(8, 0))

        # In/Out point controls
        inout_frame = ttk.Frame(right_panel, style="Dark.TFrame")
        inout_frame.pack(fill=tk.X, pady=(0, 4))

        self.in_btn = ttk.Button(inout_frame, text="Set In", command=self._set_in_point, width=7)
        self.in_btn.pack(side=tk.LEFT)
        self.out_btn = ttk.Button(inout_frame, text="Set Out", command=self._set_out_point, width=7)
        self.out_btn.pack(side=tk.LEFT, padx=(4, 0))
        self.clear_inout_btn = ttk.Button(inout_frame, text="Clear", command=self._clear_inout, width=5)
        self.clear_inout_btn.pack(side=tk.LEFT, padx=(4, 0))

        self.inout_label = ttk.Label(right_panel, text="In/Out: not set", style="Info.TLabel")
        self.inout_label.pack(anchor="w", pady=(0, 4))

        # Frame step buttons
        step_frame = ttk.Frame(right_panel, style="Dark.TFrame")
        step_frame.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(step_frame, text="◀◀", command=lambda: self._step_frame(-10), width=4).pack(side=tk.LEFT)
        ttk.Button(step_frame, text="◀", command=lambda: self._step_frame(-1), width=4).pack(side=tk.LEFT, padx=(2, 0))
        ttk.Button(step_frame, text="▶", command=lambda: self._step_frame(1), width=4).pack(side=tk.LEFT, padx=(2, 0))
        ttk.Button(step_frame, text="▶▶", command=lambda: self._step_frame(10), width=4).pack(side=tk.LEFT, padx=(2, 0))

        # Playback state
        self._playing = False
        self._play_after_id = None
        self._in_point = None   # magnitudes index
        self._out_point = None  # magnitudes index

        # Chart canvas
        self.canvas = tk.Canvas(left_panel, bg="#222222", height=180,
                                 highlightthickness=0)
        self.canvas.pack(fill=tk.X, pady=(0, 4))

        # Zoom range slider (draggable bar with handles)
        self.zoom_slider = tk.Canvas(left_panel, bg="#1a1a1a", height=24,
                                      highlightthickness=0, cursor="hand2")
        self.zoom_slider.pack(fill=tk.X, pady=(0, 4))
        self.zoom_slider.bind("<Configure>", lambda e: self._draw_zoom_slider())
        self.zoom_slider.bind("<ButtonPress-1>", self._zoom_slider_press)
        self.zoom_slider.bind("<B1-Motion>", self._zoom_slider_drag)
        self.zoom_slider.bind("<ButtonRelease-1>", self._zoom_slider_release)
        self.zoom_slider.bind("<Double-Button-1>", lambda e: self._zoom_fit())
        self._zoom_drag_mode = None  # "left", "right", "bar", or None
        self._zoom_drag_start_x = 0
        self._zoom_drag_start_vals = (0, 0)

        # Zoom state: visible range of magnitudes indices
        self._zoom_start = 0
        self._zoom_end = None  # None = show all

        # Bind chart interaction for frame cursor
        self.canvas.bind("<Motion>", self._on_chart_hover)
        self.canvas.bind("<Button-1>", self._on_chart_click)
        self.canvas.bind("<MouseWheel>", self._on_chart_scroll)
        self._cursor_line = None
        self._cursor_locked = False  # True when user clicks to lock cursor
        self._cursor_frame_idx = None
        self._in_line = None
        self._out_line = None

        # Mini-graph area for recursive passes
        self.mini_graph_frame = ttk.Frame(left_panel, style="Dark.TFrame")
        self.mini_graph_frame.pack(fill=tk.X, pady=(0, 4))

        # Editable spike table
        spike_table_frame = ttk.Frame(left_panel, style="Dark.TFrame")
        spike_table_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        # Header row
        header = ttk.Frame(spike_table_frame, style="Dark.TFrame")
        header.pack(fill=tk.X)
        for col, w in [("Type", 5), ("Frame", 6), ("Motion", 8), ("Local Avg", 8), ("Ratio", 6), ("Fix", 6)]:
            ttk.Label(header, text=col, style="Dark.TLabel", width=w,
                       font=("Consolas", 9, "bold")).pack(side=tk.LEFT, padx=2)

        # Scrollable spike rows
        spike_canvas = tk.Canvas(spike_table_frame, bg="#1a1a1a", highlightthickness=0, height=150)
        spike_scrollbar = ttk.Scrollbar(spike_table_frame, orient=tk.VERTICAL, command=spike_canvas.yview)
        self.spike_list_frame = ttk.Frame(spike_canvas, style="Dark.TFrame")
        self.spike_list_frame.bind("<Configure>",
            lambda e: spike_canvas.configure(scrollregion=spike_canvas.bbox("all")))
        spike_canvas.create_window((0, 0), window=self.spike_list_frame, anchor="nw")
        spike_canvas.configure(yscrollcommand=spike_scrollbar.set)
        spike_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        spike_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            spike_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        spike_canvas.bind("<MouseWheel>", _on_mousewheel)
        self.spike_list_frame.bind("<MouseWheel>", _on_mousewheel)

        # Row 1: Set All to Auto + Set All to N
        bulk_row = ttk.Frame(spike_table_frame, style="Dark.TFrame")
        bulk_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(bulk_row, text="Set All to Auto", command=self.set_all_auto).pack(side=tk.LEFT)
        ttk.Label(bulk_row, text="  Set All to:", style="Dark.TLabel").pack(side=tk.LEFT)
        self.bulk_count_var = tk.IntVar(value=1)
        ttk.Spinbox(bulk_row, from_=0, to=10, increment=1,
                     textvariable=self.bulk_count_var, width=4).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(bulk_row, text="Apply", command=self.set_all_manual).pack(side=tk.LEFT, padx=(4, 0))

        # Row 2: In/Out range controls — disable/enable fixes in a range
        range_row = ttk.Frame(spike_table_frame, style="Dark.TFrame")
        range_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(range_row, text="Disable In–Out", command=self._disable_inout_range).pack(side=tk.LEFT)
        ttk.Button(range_row, text="Enable In–Out", command=self._enable_inout_range).pack(side=tk.LEFT, padx=(4, 0))
        range_hint = ttk.Label(range_row, text="?", style="Dark.TLabel",
                                font=("Segoe UI", 9, "bold"), cursor="question_arrow")
        range_hint.pack(side=tk.LEFT, padx=(4, 0))
        self._add_tooltip(range_hint, "Set In and Out points on the chart, then:\n"
                                       "• Disable In–Out → sets all fixes in that range to 0\n"
                                       "• Enable In–Out → restores them to auto values\n"
                                       "Use this to ignore noisy sections of the video.")

        # Row 3: Add custom frame with type selector
        add_frame_row = ttk.Frame(spike_table_frame, style="Dark.TFrame")
        add_frame_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(add_frame_row, text="Add frame:", style="Dark.TLabel").pack(side=tk.LEFT)
        self.add_frame_entry = ttk.Entry(add_frame_row, width=6)
        self.add_frame_entry.pack(side=tk.LEFT, padx=(4, 4))
        self.add_frame_type_var = tk.StringVar(value="Spike")
        ttk.Combobox(add_frame_row, textvariable=self.add_frame_type_var,
                      values=["Spike", "Dip"], state="readonly", width=6).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(add_frame_row, text="Add", command=self.add_custom_frame).pack(side=tk.LEFT)
        add_hint = ttk.Label(add_frame_row, text="?", style="Dark.TLabel",
                              font=("Segoe UI", 9, "bold"), cursor="question_arrow")
        add_hint.pack(side=tk.LEFT, padx=(4, 0))
        self._add_tooltip(add_hint, "Manually add a frame. Choose Spike (insert frames)\n"
                                     "or Dip (remove duplicate). I/O keys set In/Out points.")

        # Storage for editable spike data
        self.spike_entries = {}  # frame_num -> tk.IntVar
        self.spike_auto_counts = {}  # frame_num -> auto-detected count (for reset)

        # Log text (compact)
        self.log = tk.Text(left_panel, bg="#222222", fg="#cccccc",
                           font=("Consolas", 9), height=4, wrap=tk.WORD,
                           insertbackground="#cccccc", relief=tk.FLAT)
        self.log.pack(fill=tk.X)

        # Tag for spike highlights
        self.log.tag_configure("spike", foreground="#ff6b6b", font=("Consolas", 9, "bold"))
        self.log.tag_configure("dip", foreground="#ffcc44", font=("Consolas", 9, "bold"))
        self.log.tag_configure("good", foreground="#6bff6b")
        self.log.tag_configure("header", foreground="#ffffff", font=("Consolas", 10, "bold"))

        # Keyboard shortcuts
        self.root.bind("<i>", lambda e: self._set_in_point() if not isinstance(e.widget, (tk.Entry, ttk.Entry, ttk.Spinbox)) else None)
        self.root.bind("<o>", lambda e: self._set_out_point() if not isinstance(e.widget, (tk.Entry, ttk.Entry, ttk.Spinbox)) else None)
        self.root.bind("<I>", lambda e: self._set_in_point() if not isinstance(e.widget, (tk.Entry, ttk.Entry, ttk.Spinbox)) else None)
        self.root.bind("<O>", lambda e: self._set_out_point() if not isinstance(e.widget, (tk.Entry, ttk.Entry, ttk.Spinbox)) else None)
        self.root.bind("<space>", lambda e: self._toggle_play() if not isinstance(e.widget, (tk.Entry, ttk.Entry, ttk.Spinbox)) else None)

        # Enable drag & drop via simple protocol
        self.setup_drop_zone()

    def _add_tooltip(self, widget, text):
        """Add a hover tooltip to any widget."""
        tip = None

        def show(event):
            nonlocal tip
            tip = tk.Toplevel(widget)
            tip.wm_overrideredirect(True)
            tip.wm_geometry(f"+{event.x_root + 12}+{event.y_root + 8}")
            tip.configure(bg="#333333")
            label = tk.Label(tip, text=text, bg="#333333", fg="#e0e0e0",
                             font=("Segoe UI", 9), justify=tk.LEFT, padx=8, pady=6,
                             wraplength=320)
            label.pack()

        def hide(event):
            nonlocal tip
            if tip:
                tip.destroy()
                tip = None

        widget.bind("<Enter>", show)
        widget.bind("<Leave>", hide)

    def setup_drop_zone(self):
        """Make the window accept files via drag (basic tkinter approach)."""
        # tkinter doesn't natively support drag-and-drop from Explorer,
        # but we can detect it if tkinterdnd2 is available
        try:
            from tkinterdnd2 import DND_FILES
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self.on_drop)
        except ImportError:
            pass  # No drag-drop support, browse button still works

    def on_drop(self, event):
        path = event.data.strip('{}')
        if path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
            self.video_path = Path(path)
            self.file_label.configure(text=str(self.video_path))
        else:
            messagebox.showwarning("Invalid file", "Please drop a video file.")

    def browse_file(self):
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("All files", "*.*"), ("Video files", "*.mp4 *.mov *.avi *.mkv *.webm *.mxf *.ts")]
        )
        if path:
            self.video_path = Path(path)
            self.file_label.configure(text=str(self.video_path))
            self.log_msg(f"Loaded: {self.video_path.name}\n")

    def log_msg(self, msg, tag=None):
        self.log.insert(tk.END, msg, tag)
        self.log.see(tk.END)

    def _log_clickable_path(self, filepath):
        """Log a file path that opens the containing folder when clicked."""
        import os
        tag_name = f"link_{id(filepath)}"
        self.log.tag_configure(tag_name, foreground="#66bbff", underline=True,
                                font=("Consolas", 9))
        self.log.tag_bind(tag_name, "<Button-1>",
                           lambda e, p=filepath: self._open_in_explorer(p))
        self.log.tag_bind(tag_name, "<Enter>",
                           lambda e: self.log.configure(cursor="hand2"))
        self.log.tag_bind(tag_name, "<Leave>",
                           lambda e: self.log.configure(cursor=""))
        self.log.insert(tk.END, f"Saved to: {filepath}\n", tag_name)
        self.log.see(tk.END)

    def _open_in_explorer(self, filepath):
        """Open Explorer with the file selected."""
        import os, platform
        filepath = os.path.normpath(filepath)
        if platform.system() == "Windows":
            subprocess.Popen(["explorer", "/select,", filepath])
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", "-R", filepath])
        else:
            subprocess.Popen(["xdg-open", os.path.dirname(filepath)])

    def set_status(self, msg):
        self.status_label.configure(text=msg)

    def start_analyze(self):
        if not self.video_path or not self.video_path.exists():
            messagebox.showwarning("No file", "Please select a video file first.")
            return
        if self.analyzing:
            return

        self.analyzing = True
        self.analyze_btn.configure(state=tk.DISABLED)
        self.fix_btn.configure(state=tk.DISABLED)
        self.export_btn.configure(state=tk.DISABLED)
        self.log.delete("1.0", tk.END)
        self.canvas.delete("all")

        thread = threading.Thread(target=self.run_analysis, daemon=True)
        thread.start()

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

            info = f"Video: {self.width}x{self.height} @ {self.fps:.2f}fps, {total} frames\n"
            self.root.after(0, lambda: self.log_msg(info, "header"))

            ret, prev_frame = cap.read()
            if not ret:
                return

            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            self.magnitudes = []
            self.frames_data = [(0, prev_frame.copy())]

            frame_idx = 1
            while True:
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
                self.frames_data.append((frame_idx, frame.copy()))

                # Update progress
                pct = (frame_idx / total) * 100
                self.root.after(0, lambda p=pct, f=frame_idx: (
                    self.progress_var.set(p),
                    self.set_status(f"Analyzing frame {f}/{total}...")
                ))

                prev_gray = gray
                frame_idx += 1

            cap.release()

            # Detect spikes and dips
            threshold = self.threshold_var.get()
            self.spikes = self.detect_spikes(self.magnitudes, threshold)
            self.dips = self.detect_dips(self.magnitudes, threshold)

            # Recalculate spike est_missing with dips excluded from median
            # (same spikes detected, just cleaner frame count estimates)
            dip_frame_set = set(d['frame'] for d in self.dips)
            if dip_frame_set:
                self._recalc_spike_estimates(dip_frame_set)

            # Update UI on main thread
            self.root.after(0, self.show_results)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, lambda: (
                self.analyze_btn.configure(state=tk.NORMAL),
                setattr(self, 'analyzing', False)
            ))

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
                    est_missing = max(1, round(ratio) - 1)
                    spikes.append({
                        'frame': magnitudes[i][0],
                        'magnitude': float(mags[i]),
                        'local_median': local_median,
                        'ratio': float(ratio),
                        'est_missing': est_missing
                    })
        return spikes

    def _recalc_spike_estimates(self, dip_frame_set):
        """Recalculate est_missing for detected spikes, excluding dip frames
        from the local median. Same spikes, just better frame count estimates."""
        mags = np.array([m[1] for m in self.magnitudes])
        mag_frames = [m[0] for m in self.magnitudes]
        window = min(15, len(mags) // 2)
        if window < 3:
            window = 3

        for spike in self.spikes:
            # Find this spike's index in magnitudes
            try:
                i = mag_frames.index(spike['frame'])
            except ValueError:
                continue

            start = max(0, i - window)
            end = min(len(mags), i + window + 1)

            # Build local neighborhood excluding dips
            neighbors = []
            for j in range(start, end):
                if j != i and mag_frames[j] not in dip_frame_set:
                    neighbors.append(mags[j])

            if not neighbors:
                continue

            local_median = float(np.median(neighbors))
            if local_median > 0:
                ratio = mags[i] / local_median
                spike['local_median'] = local_median
                spike['ratio'] = float(ratio)
                spike['est_missing'] = max(1, round(ratio) - 1)

    def detect_dips(self, magnitudes, threshold_multiplier=2.0):
        """Detect frames with abnormally LOW motion (duplicate/stuck frames)."""
        if len(magnitudes) < 5:
            return []

        mags = np.array([m[1] for m in magnitudes])
        # Use a smaller window for dips — tight neighbors give a better
        # sense of what's "normal" right around the frame.
        window = min(7, len(mags) // 2)
        if window < 3:
            window = 3

        dips = []
        # Dips need more sensitivity than a simple inverse.
        # At spike threshold 2.0 → dip threshold 0.70 (catches frames at ≤70% of median)
        # At spike threshold 1.5 → dip threshold 0.80
        # Lower spike threshold = catches more of both.
        dip_threshold = min(0.80, 1.0 / threshold_multiplier + 0.20)

        # Skip first 2 and last 2 frames — edge frames have unreliable motion
        for i in range(2, len(mags) - 2):
            start = max(0, i - window)
            end = min(len(mags), i + window + 1)
            # Exclude spikes from the local neighborhood so they don't
            # inflate the median and cause false dip detections nearby
            spike_frame_set = set(s['frame'] for s in self.spikes) if hasattr(self, 'spikes') else set()
            neighbors = []
            for j in range(start, end):
                if j != i and magnitudes[j][0] not in spike_frame_set:
                    neighbors.append(mags[j])
            if not neighbors:
                # Fallback: use all neighbors if filtering removed everything
                local = np.concatenate([mags[start:i], mags[i+1:end]])
            else:
                local = np.array(neighbors)
            if len(local) == 0:
                continue

            local_median = float(np.median(local))

            if local_median > 0:
                ratio = mags[i] / local_median
                # Flag if motion drops significantly below neighbors
                if ratio < dip_threshold:
                    dips.append({
                        'frame': magnitudes[i][0],
                        'magnitude': float(mags[i]),
                        'local_median': local_median,
                        'ratio': float(ratio),
                    })
        return dips

    def show_results(self):
        self.progress_var.set(100)
        self.draw_chart()

        # Clear old spike table rows
        for w in self.spike_list_frame.winfo_children():
            w.destroy()
        self.spike_entries = {}
        self.spike_auto_counts = {}
        self.dip_entries = set()  # track which frame_nums are dips

        self.log_msg(f"\nAnalysis complete: {len(self.magnitudes)} frames\n", "header")

        has_issues = False

        if self.spikes:
            self.log_msg(f"Found {len(self.spikes)} spikes (missing frames) — shown in red.\n", "spike")
            has_issues = True
        if self.dips:
            self.log_msg(f"Found {len(self.dips)} dips (duplicate frames) — shown in yellow.\n", "dip")
            has_issues = True

        if has_issues:
            # Merge spikes and dips into one sorted list by frame number
            all_issues = []
            for s in self.spikes:
                all_issues.append(('spike', s['frame'], s['magnitude'], s['local_median'],
                                    s['ratio'], s['est_missing']))
            for d in self.dips:
                all_issues.append(('dip', d['frame'], d['magnitude'], d['local_median'],
                                    d['ratio'], 1))
            all_issues.sort(key=lambda x: x[1])  # sort by frame number

            for kind, frame, mag, local_avg, ratio, count in all_issues:
                self._add_spike_row(frame, mag, local_avg, ratio, count, kind=kind)

            self.log_msg("Edit table below, then Fix. Set 0 to skip.\n")
            self.fix_btn.configure(state=tk.NORMAL)
            self.recursive_btn.configure(state=tk.NORMAL)
            self.export_btn.configure(state=tk.NORMAL)
            self.set_status(f"Found {len(self.spikes)} spikes, {len(self.dips)} dips. Edit, then Fix.")
        else:
            self.log_msg("No issues detected — video looks clean!\n", "good")
            self.export_btn.configure(state=tk.NORMAL)
            self.set_status("No issues found.")

    def _add_spike_row(self, frame_num, motion, local_avg, ratio, insert_count, kind="spike"):
        """Add an editable row to the spike table. kind = 'spike' or 'dip'."""
        row = ttk.Frame(self.spike_list_frame, style="Dark.TFrame")
        row.pack(fill=tk.X, pady=1)

        # Type indicator
        if kind == "spike":
            type_text = "\u25b2 Spike"
            type_fg = "#ff6b6b"
        else:
            type_text = "\u25bc Dip"
            type_fg = "#ffcc44"

        type_lbl = tk.Label(row, text=type_text, bg="#1a1a1a", fg=type_fg,
                             font=("Consolas", 9, "bold"), width=7, anchor="w")
        type_lbl.pack(side=tk.LEFT, padx=2)

        ttk.Label(row, text=f"{frame_num}", style="Dark.TLabel", width=6,
                   font=("Consolas", 9)).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text=f"{motion:.2f}", style="Dark.TLabel", width=8,
                   font=("Consolas", 9)).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text=f"{local_avg:.2f}", style="Dark.TLabel", width=8,
                   font=("Consolas", 9)).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text=f"{ratio:.2f}x", style="Dark.TLabel", width=6,
                   font=("Consolas", 9)).pack(side=tk.LEFT, padx=2)

        # For spikes: spinbox controls how many frames to insert (0-10)
        # For dips: spinbox is 0 (skip) or 1 (replace) only
        var = tk.IntVar(value=insert_count)
        max_val = 1 if kind == "dip" else 10
        spin = ttk.Spinbox(row, from_=0, to=max_val, increment=1, textvariable=var, width=4)
        spin.pack(side=tk.LEFT, padx=2)

        # ✕ button sets to 0 (disables) instead of removing the row
        def disable_row(f=frame_num, v=var):
            v.set(0)
            self.draw_chart()  # redraw to show gray
        ttk.Button(row, text="\u2715", width=2, command=disable_row).pack(side=tk.LEFT, padx=(4, 0))

        # Hover row → move chart cursor to this frame
        def _on_row_enter(event, f=frame_num):
            if not self._cursor_locked and self.magnitudes:
                # Find the magnitudes index for this frame number
                for mi, (fn, _) in enumerate(self.magnitudes):
                    if fn == f:
                        self._update_cursor(mi)
                        break
        row.bind("<Enter>", _on_row_enter)
        for child in row.winfo_children():
            child.bind("<Enter>", _on_row_enter)

        # Redraw chart when value changes to update bar colors
        var.trace_add("write", lambda *_: self.draw_chart())

        self.spike_entries[frame_num] = var
        self.spike_auto_counts[frame_num] = insert_count
        if kind == "dip":
            self.dip_entries.add(frame_num)

    def add_custom_frame(self):
        """Add a manually specified frame to the spike table."""
        try:
            frame_num = int(self.add_frame_entry.get().strip())
        except ValueError:
            return

        if frame_num in self.spike_entries:
            return  # already in table

        # Check it's a valid frame
        total = len(self.frames_data)
        if frame_num < 0 or frame_num >= total:
            return

        # Determine type from selector
        kind = "dip" if self.add_frame_type_var.get() == "Dip" else "spike"

        # Save current values, add new entry, rebuild sorted
        current = {f: v.get() for f, v in self.spike_entries.items()}
        current[frame_num] = 1
        self.spike_auto_counts[frame_num] = 1
        if kind == "dip":
            self.dip_entries.add(frame_num)
        self._rebuild_spike_table(current)

        self.add_frame_entry.delete(0, tk.END)
        self.fix_btn.configure(state=tk.NORMAL)
        self.recursive_btn.configure(state=tk.NORMAL)
        tag = "dip" if kind == "dip" else "spike"
        self.log_msg(f"  Added frame {frame_num} as {kind} manually\n", tag)

    def _rebuild_spike_table(self, counts):
        """Clear and rebuild spike table rows in sorted frame order."""
        old_dips = set(self.dip_entries)  # preserve dip knowledge
        for w in self.spike_list_frame.winfo_children():
            w.destroy()
        self.spike_entries = {}
        self.dip_entries = set()

        mag_dict = {m[0]: m[1] for m in self.magnitudes}
        spike_dict = {s['frame']: s for s in self.spikes}
        dip_dict = {d['frame']: d for d in getattr(self, 'dips', [])}

        for frame_num in sorted(counts.keys()):
            # Determine if this is a dip or spike
            if frame_num in dip_dict or frame_num in old_dips:
                d = dip_dict.get(frame_num, {})
                motion = d.get('magnitude', mag_dict.get(frame_num, 0.0))
                local_avg = d.get('local_median', 0.0)
                ratio = d.get('ratio', 0.0)
                self._add_spike_row(frame_num, motion, local_avg, ratio, counts[frame_num], kind="dip")
            else:
                s = spike_dict.get(frame_num, {})
                motion = s.get('magnitude', mag_dict.get(frame_num, 0.0))
                local_avg = s.get('local_median', 0.0)
                ratio = s.get('ratio', 0.0)
                self._add_spike_row(frame_num, motion, local_avg, ratio, counts[frame_num], kind="spike")

    def set_all_auto(self):
        """Reset all spike rows to their auto-detected counts."""
        for frame_num, var in self.spike_entries.items():
            auto = self.spike_auto_counts.get(frame_num, 1)
            var.set(auto)

    def set_all_manual(self):
        """Set all spike rows to the bulk count value."""
        count = self.bulk_count_var.get()
        for var in self.spike_entries.values():
            var.set(count)

    def draw_chart(self):
        """Draw motion magnitude chart on canvas with zoom support."""
        self.canvas.delete("all")
        self._cursor_line = None
        self._in_line = None
        self._out_line = None
        c = self.canvas
        c.update_idletasks()
        cw = c.winfo_width()
        ch = c.winfo_height()

        if not self.magnitudes or cw < 10 or ch < 10:
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
            return

        # Store chart geometry for cursor interaction
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
        # Include both auto-detected and manually added entries
        spike_frames = set(s['frame'] for s in self.spikes)
        dip_frames = set(d['frame'] for d in getattr(self, 'dips', []))
        # Add manually added entries from the table
        for fn in self.spike_entries:
            if fn in getattr(self, 'dip_entries', set()):
                dip_frames.add(fn)
            elif fn not in dip_frames:
                spike_frames.add(fn)

        # Axes
        c.create_line(pad_left, pad_top, pad_left, ch - pad_bottom, fill="#555")
        c.create_line(pad_left, ch - pad_bottom, cw - pad_right, ch - pad_bottom, fill="#555")

        # Y-axis label
        c.create_text(pad_left - 5, pad_top, text=f"{max_mag:.1f}", anchor="ne",
                       fill="#888", font=("Consolas", 8))
        c.create_text(pad_left - 5, ch - pad_bottom, text="0", anchor="ne",
                       fill="#888", font=("Consolas", 8))

        # X-axis labels (frame numbers)
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

                # Check if this frame is disabled (set to 0 in table)
                is_disabled = False
                if frame_num in self.spike_entries:
                    try:
                        is_disabled = self.spike_entries[frame_num].get() == 0
                    except (tk.TclError, ValueError):
                        pass

                if frame_num in spike_frames:
                    if is_disabled:
                        bar_color = "#555555"
                        label_color = "#777777"
                    else:
                        bar_color = "#ff4444"
                        label_color = "#ff6b6b"
                    c.create_rectangle(x, y, x + max(bar_w, 2), ch - pad_bottom,
                                        fill=bar_color, outline="")
                    if n_vis < 200:
                        c.create_text(x, y - 8, text=f"F{frame_num}", fill=label_color,
                                       font=("Consolas", 7), anchor="s")
                elif frame_num in dip_frames:
                    if is_disabled:
                        bar_color = "#555555"
                        label_color = "#777777"
                    else:
                        bar_color = "#ccaa00"
                        label_color = "#ffcc44"
                    c.create_rectangle(x, y, x + max(bar_w, 2), ch - pad_bottom,
                                        fill=bar_color, outline="")
                    if n_vis < 200:
                        c.create_text(x, y - 8, text=f"F{frame_num}", fill=label_color,
                                       font=("Consolas", 7), anchor="s")
                else:
                    c.create_rectangle(x, y, x + max(bar_w, 2), ch - pad_bottom,
                                        fill="#3a7acc", outline="")

                # Show frame numbers on x-axis when zoomed in
                if n_vis <= 60 and vi % max(1, n_vis // 15) == 0:
                    c.create_text(x, ch - pad_bottom + 12, text=str(frame_num),
                                   anchor="n", fill="#666", font=("Consolas", 7))

        # Draw in/out markers
        self._draw_inout_markers()

        # Title
        zoom_info = f"  [{first_frame}–{last_frame}]" if zs > 0 or ze < len(self.magnitudes) else ""
        c.create_text(cw // 2, 8, text=f"Motion Magnitude per Frame{zoom_info}  (click to lock)",
                       fill="#aaa", font=("Segoe UI", 9), anchor="n")

        # Update zoom slider to match
        self._draw_zoom_slider()

    # ── Chart cursor & frame preview ──────────────────────────────────

    def _chart_x_to_frame_index(self, x):
        """Convert canvas x-coordinate to magnitudes list index (global)."""
        g = getattr(self, '_chart_geom', None)
        if not g or not self.magnitudes:
            return None
        rel = x - g['pad_left']
        if rel < 0 or rel > g['plot_w']:
            return None
        vi = int((rel / g['plot_w']) * g['n'])
        vi = max(0, min(vi, g['n'] - 1))
        return g['zoom_start'] + vi  # return global index

    def _global_idx_to_chart_x(self, idx):
        """Convert global magnitudes index to canvas x-coordinate."""
        g = getattr(self, '_chart_geom', None)
        if not g:
            return None
        vi = idx - g['zoom_start']
        if vi < 0 or vi >= g['n']:
            return None
        return g['pad_left'] + (vi / g['n']) * g['plot_w']

    def _on_chart_hover(self, event):
        """Show cursor line on chart hover (unless locked)."""
        if self._cursor_locked:
            return
        self._update_cursor_from_x(event.x)

    def _on_chart_click(self, event):
        """Click to lock cursor, click again to unlock."""
        if self._cursor_locked:
            self._cursor_locked = False
            self._update_cursor_from_x(event.x)
        else:
            self._cursor_locked = True
            self._update_cursor_from_x(event.x)

    def _on_chart_scroll(self, event):
        """Scroll wheel on chart to zoom in/out centered on cursor."""
        if not self.magnitudes:
            return
        g = getattr(self, '_chart_geom', None)
        if not g:
            return

        # Figure out center frame from mouse position
        center_idx = self._chart_x_to_frame_index(event.x)
        if center_idx is None:
            center_idx = (g['zoom_start'] + g['zoom_end']) // 2

        total = len(self.magnitudes)
        current_span = g['zoom_end'] - g['zoom_start']

        if event.delta > 0:
            # Zoom in
            new_span = max(20, int(current_span * 0.7))
        else:
            # Zoom out
            new_span = min(total, int(current_span * 1.4))

        half = new_span // 2
        self._zoom_start = max(0, center_idx - half)
        self._zoom_end = min(total, self._zoom_start + new_span)
        if self._zoom_end >= total:
            self._zoom_start = max(0, total - new_span)
            self._zoom_end = total

        self.draw_chart()
        self._draw_zoom_slider()

    def _update_cursor_from_x(self, x):
        """Update cursor from canvas x-coordinate."""
        idx = self._chart_x_to_frame_index(x)
        if idx is not None:
            self._update_cursor(idx)

    def _update_cursor(self, idx):
        """Draw cursor line and update frame preview. idx = global magnitudes index."""
        g = getattr(self, '_chart_geom', None)
        if not g or not self.magnitudes:
            return

        idx = max(0, min(idx, len(self.magnitudes) - 1))

        # Avoid redundant redraws
        if idx == self._cursor_frame_idx:
            return
        self._cursor_frame_idx = idx

        c = self.canvas

        # Remove old cursor line
        if self._cursor_line:
            c.delete(self._cursor_line)
            self._cursor_line = None

        # Draw new cursor line (only if visible in zoom range)
        snap_x = self._global_idx_to_chart_x(idx)
        if snap_x is not None:
            self._cursor_line = c.create_line(
                snap_x, g['pad_top'], snap_x, g['ch'] - g['pad_bottom'],
                fill="#ffdd44", width=2)

        # Update frame preview
        frame_num = self.magnitudes[idx][0]
        mag_val = self.magnitudes[idx][1]
        issue_tag = ""
        if frame_num in {s['frame'] for s in self.spikes}:
            issue_tag = "  \u26a0 SPIKE"
        elif frame_num in {d['frame'] for d in getattr(self, 'dips', [])}:
            issue_tag = "  \u25bc DIP"
        lock_tag = " \U0001f512" if self._cursor_locked else ""
        self.preview_label.configure(
            text=f"Frame {frame_num}  |  Motion: {mag_val:.2f}{issue_tag}{lock_tag}")
        self._show_frame_preview(idx)

    def _show_frame_preview(self, mag_idx):
        """Display the frame at magnitudes index in the preview canvas."""
        if not self.frames_data:
            return

        # magnitudes index is offset by 1 from frames_data index
        # (magnitudes[0] is frame 1, frames_data[0] is frame 0)
        data_idx = mag_idx + 1  # +1 because magnitudes start from frame 1
        if data_idx < 0 or data_idx >= len(self.frames_data):
            data_idx = mag_idx

        _, bgr_frame = self.frames_data[data_idx]

        # Convert BGR to RGB for tkinter
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # Fit into preview canvas
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

        # Convert to PhotoImage
        from PIL import Image, ImageTk
        img = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(img)

        pc.delete("all")
        pc.create_image(pw // 2, ph // 2, image=photo, anchor="center")
        self._preview_photo = photo  # prevent GC

    # ── Zoom range slider ────────────────────────────────────────────

    def _draw_zoom_slider(self):
        """Draw the zoom range slider bar with draggable handles."""
        zs = self.zoom_slider
        zs.delete("all")
        zs.update_idletasks()
        w = zs.winfo_width()
        h = zs.winfo_height()
        if w < 20 or not self.magnitudes:
            return

        total = len(self.magnitudes)
        pad = 10  # horizontal padding
        track_w = w - pad * 2
        bar_y = h // 2
        handle_r = 7

        # Track background
        zs.create_rectangle(pad, bar_y - 3, w - pad, bar_y + 3,
                             fill="#333333", outline="")

        # Compute handle positions from zoom state
        z_start = self._zoom_start
        z_end = self._zoom_end if self._zoom_end is not None else total
        left_x = pad + (z_start / total) * track_w
        right_x = pad + (z_end / total) * track_w

        # Active range bar
        zs.create_rectangle(left_x, bar_y - 5, right_x, bar_y + 5,
                             fill="#4a9eff", outline="", tags="bar")

        # Left handle
        zs.create_oval(left_x - handle_r, bar_y - handle_r,
                        left_x + handle_r, bar_y + handle_r,
                        fill="#ffffff", outline="#888888", width=1, tags="left_handle")

        # Right handle
        zs.create_oval(right_x - handle_r, bar_y - handle_r,
                        right_x + handle_r, bar_y + handle_r,
                        fill="#ffffff", outline="#888888", width=1, tags="right_handle")

        # Store layout for drag calculations
        self._zoom_slider_geom = {
            'pad': pad, 'track_w': track_w, 'total': total,
            'left_x': left_x, 'right_x': right_x, 'handle_r': handle_r
        }

    def _zoom_slider_press(self, event):
        """Determine what the user grabbed: left handle, right handle, or bar."""
        sg = getattr(self, '_zoom_slider_geom', None)
        if not sg:
            return

        x = event.x
        hr = sg['handle_r'] + 4  # slightly generous hit area

        if abs(x - sg['left_x']) <= hr:
            self._zoom_drag_mode = "left"
        elif abs(x - sg['right_x']) <= hr:
            self._zoom_drag_mode = "right"
        elif sg['left_x'] < x < sg['right_x']:
            self._zoom_drag_mode = "bar"
        else:
            # Clicked outside — jump the bar center to this position
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
            self._draw_zoom_slider()

        self._zoom_drag_start_x = x
        self._zoom_drag_start_vals = (self._zoom_start,
                                       self._zoom_end or len(self.magnitudes))

    def _zoom_slider_drag(self, event):
        """Handle drag on the zoom slider."""
        sg = getattr(self, '_zoom_slider_geom', None)
        if not sg or not self._zoom_drag_mode:
            return

        total = sg['total']
        dx_pixels = event.x - self._zoom_drag_start_x
        dx_frames = int((dx_pixels / sg['track_w']) * total)
        orig_start, orig_end = self._zoom_drag_start_vals
        min_span = 10  # minimum visible frames

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
        self._draw_zoom_slider()

    def _zoom_slider_release(self, event):
        """End drag."""
        self._zoom_drag_mode = None

    def _zoom_fit(self):
        """Reset zoom to show all frames (double-click on slider)."""
        self._zoom_start = 0
        self._zoom_end = None
        self.draw_chart()
        self._draw_zoom_slider()

    # ── Playback controls ──────────────────────────────────────────────

    def _toggle_play(self):
        """Play / Pause toggle."""
        if self._playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        if not self.frames_data or not self.magnitudes:
            return
        self._playing = True
        self.play_btn.configure(text="⏸ Pause")
        self._cursor_locked = True

        # Start from current cursor, or in-point, or beginning
        if self._cursor_frame_idx is None:
            start = self._in_point if self._in_point is not None else 0
            self._cursor_frame_idx = start

        self._play_next_frame()

    def _stop_playback(self):
        self._playing = False
        self.play_btn.configure(text="▶ Play")
        if self._play_after_id:
            self.root.after_cancel(self._play_after_id)
            self._play_after_id = None

    def _play_next_frame(self):
        if not self._playing or not self.magnitudes:
            return

        idx = self._cursor_frame_idx
        if idx is None:
            idx = 0

        # Determine play range
        start = self._in_point if self._in_point is not None else 0
        end = self._out_point if self._out_point is not None else len(self.magnitudes) - 1

        # Advance
        idx += 1
        if idx > end:
            if self.loop_var.get():
                idx = start
            else:
                self._stop_playback()
                return

        self._cursor_frame_idx = None  # reset to force redraw
        self._update_cursor(idx)

        # Auto-scroll chart to follow playhead
        g = getattr(self, '_chart_geom', None)
        if g and (idx < g['zoom_start'] or idx >= g['zoom_end']):
            span = g['zoom_end'] - g['zoom_start']
            self._zoom_start = max(0, idx - span // 4)
            self._zoom_end = self._zoom_start + span
            self.draw_chart()

        # Schedule next frame based on speed
        speed_str = self.speed_var.get()
        speed = float(speed_str.replace('x', ''))
        if self.fps > 0:
            delay_ms = max(1, int(1000 / (self.fps * speed)))
        else:
            delay_ms = 42  # ~24fps default

        self._play_after_id = self.root.after(delay_ms, self._play_next_frame)

    def _step_frame(self, delta):
        """Step forward/backward by delta frames."""
        if not self.magnitudes:
            return
        idx = self._cursor_frame_idx if self._cursor_frame_idx is not None else 0
        idx = max(0, min(len(self.magnitudes) - 1, idx + delta))
        self._cursor_locked = True
        self._cursor_frame_idx = None  # force redraw
        self._update_cursor(idx)

        # Auto-scroll if needed
        g = getattr(self, '_chart_geom', None)
        if g and (idx < g['zoom_start'] or idx >= g['zoom_end']):
            span = g['zoom_end'] - g['zoom_start']
            self._zoom_start = max(0, idx - span // 4)
            self._zoom_end = self._zoom_start + span
            self.draw_chart()

    # ── In/Out points ──────────────────────────────────────────────────

    def _set_in_point(self):
        """Set in-point at current cursor position."""
        if self._cursor_frame_idx is not None:
            self._in_point = self._cursor_frame_idx
            self._update_inout_label()
            self.draw_chart()

    def _set_out_point(self):
        """Set out-point at current cursor position."""
        if self._cursor_frame_idx is not None:
            self._out_point = self._cursor_frame_idx
            self._update_inout_label()
            self.draw_chart()

    def _clear_inout(self):
        """Clear in/out points."""
        self._in_point = None
        self._out_point = None
        self._update_inout_label()
        self.draw_chart()

    def _get_inout_frame_range(self):
        """Get the frame number range from in/out points."""
        if self._in_point is None or self._out_point is None or not self.magnitudes:
            return None, None
        in_frame = self.magnitudes[min(self._in_point, self._out_point)][0]
        out_frame = self.magnitudes[max(self._in_point, self._out_point)][0]
        return in_frame, out_frame

    def _disable_inout_range(self):
        """Set all fixes in the In–Out range to 0."""
        in_f, out_f = self._get_inout_frame_range()
        if in_f is None:
            messagebox.showinfo("No range", "Set In and Out points on the chart first.")
            return
        count = 0
        for frame_num, var in self.spike_entries.items():
            if in_f <= frame_num <= out_f:
                var.set(0)
                count += 1
        self.log_msg(f"  Disabled {count} fixes in range F{in_f}–F{out_f}\n")
        self.draw_chart()

    def _enable_inout_range(self):
        """Restore all fixes in the In–Out range to their auto values."""
        in_f, out_f = self._get_inout_frame_range()
        if in_f is None:
            messagebox.showinfo("No range", "Set In and Out points on the chart first.")
            return
        count = 0
        for frame_num, var in self.spike_entries.items():
            if in_f <= frame_num <= out_f:
                auto = self.spike_auto_counts.get(frame_num, 1)
                var.set(auto)
                count += 1
        self.log_msg(f"  Enabled {count} fixes in range F{in_f}–F{out_f}\n")
        self.draw_chart()

    def _update_inout_label(self):
        in_str = f"F{self.magnitudes[self._in_point][0]}" if self._in_point is not None and self.magnitudes else "—"
        out_str = f"F{self.magnitudes[self._out_point][0]}" if self._out_point is not None and self.magnitudes else "—"
        self.inout_label.configure(text=f"In: {in_str}  |  Out: {out_str}")

    def _draw_inout_markers(self):
        """Draw in/out point lines on the chart."""
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

    def _init_raft(self):
        """Lazy-load RAFT model on first use."""
        if not hasattr(self, '_raft_model'):
            import torch
            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            weights = Raft_Large_Weights.DEFAULT
            self._raft_model = raft_large(weights=weights).to(self._device).eval()
            self._raft_transforms = weights.transforms()
            print(f"RAFT loaded on {self._device}")

    def _compute_flow_raft(self, frame0, frame1):
        """Compute optical flow using RAFT neural network — handles large motion."""
        import torch
        self._init_raft()

        # Convert BGR→RGB, to tensor, normalize
        img0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
        img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

        t0 = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(0).float().to(self._device)
        t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().to(self._device)

        # RAFT needs dimensions divisible by 8 — pad if needed
        h, w = t0.shape[2], t0.shape[3]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            t0 = torch.nn.functional.pad(t0, [0, pad_w, 0, pad_h], mode='replicate')
            t1 = torch.nn.functional.pad(t1, [0, pad_w, 0, pad_h], mode='replicate')

        with torch.no_grad():
            # RAFT returns list of flow predictions at different iterations
            flow_list = self._raft_model(t0, t1)
            flow = flow_list[-1]  # Final (best) prediction

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            flow = flow[:, :, :h, :w]

        return flow  # [1, 2, H, W] tensor on GPU

    def interpolate_frame(self, frame_before, frame_after, alpha=0.5):
        """Generate an in-between frame using RAFT neural optical flow.

        RAFT handles large displacements (missing frames = big jumps) far better
        than traditional optical flow. Uses bidirectional flow + occlusion-aware
        blending for clean results.
        """
        import torch
        import torch.nn.functional as F

        h, w = frame_before.shape[:2]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Compute neural optical flow (GPU, RAFT)
        flow_fwd = self._compute_flow_raft(frame_before, frame_after)  # [1, 2, H, W]
        flow_bwd = self._compute_flow_raft(frame_after, frame_before)

        # Prepare image tensors
        img0 = torch.from_numpy(frame_before.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        img1 = torch.from_numpy(frame_after.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

        # Scale flow to normalized coordinates for grid_sample [-1, 1]
        f01_norm = flow_fwd.clone()
        f01_norm[:, 0, :, :] = f01_norm[:, 0, :, :] / ((w - 1) / 2)
        f01_norm[:, 1, :, :] = f01_norm[:, 1, :, :] / ((h - 1) / 2)

        f10_norm = flow_bwd.clone()
        f10_norm[:, 0, :, :] = f10_norm[:, 0, :, :] / ((w - 1) / 2)
        f10_norm[:, 1, :, :] = f10_norm[:, 1, :, :] / ((h - 1) / 2)

        # Create base grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device),
            torch.linspace(-1, 1, w, device=device),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)

        # Warp img0 forward by alpha * flow
        grid_fwd = base_grid + f01_norm * alpha
        grid_fwd = grid_fwd.permute(0, 2, 3, 1)
        warped0 = F.grid_sample(img0, grid_fwd, mode='bilinear', padding_mode='border', align_corners=True)

        # Warp img1 backward by (1-alpha) * flow
        grid_bwd = base_grid + f10_norm * (1 - alpha)
        grid_bwd = grid_bwd.permute(0, 2, 3, 1)
        warped1 = F.grid_sample(img1, grid_bwd, mode='bilinear', padding_mode='border', align_corners=True)

        # Occlusion detection via forward-backward consistency
        f10_warped = F.grid_sample(f10_norm, grid_fwd, mode='bilinear', padding_mode='border', align_corners=True)
        # Consistent flow: fwd + warped_bwd should cancel out
        flow_sum = f01_norm * alpha + f10_warped * alpha
        flow_diff = torch.sum(flow_sum ** 2, dim=1, keepdim=True)

        # Soft occlusion mask
        occ_thresh = 0.01 * (torch.sum(f01_norm ** 2, dim=1, keepdim=True) +
                              torch.sum(f10_warped ** 2, dim=1, keepdim=True)) + 0.5
        occ_mask = (flow_diff > occ_thresh).float()
        occ_mask = F.avg_pool2d(occ_mask, kernel_size=5, stride=1, padding=2)  # smooth edges

        # Blend: occluded areas favor the less-warped frame
        weight0 = (1 - alpha) * (1 - occ_mask * 0.5)
        weight1 = alpha * (1 + occ_mask * 0.5)
        total = weight0 + weight1 + 1e-8
        weight0 = weight0 / total
        weight1 = weight1 / total

        blended = warped0 * weight0 + warped1 * weight1

        # Back to numpy
        result = (blended.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return result

    @staticmethod
    def _forward_splat(img, flow, t):
        """Forward-warp img by t * flow using splatting (push-based).
        Correctly handles large displacements by pushing each source pixel
        to its destination rather than pulling (which fails for big motion).
        """
        h, w = flow.shape[:2]
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)

        dest_x = x + t * flow[..., 0]
        dest_y = y + t * flow[..., 1]

        x0 = np.floor(dest_x).astype(np.int32)
        y0 = np.floor(dest_y).astype(np.int32)
        wx = dest_x - x0
        wy = dest_y - y0

        output = np.zeros((h, w, 3), dtype=np.float64)
        weight = np.zeros((h, w), dtype=np.float64)
        pixels = img.astype(np.float64)

        for dy_off in [0, 1]:
            for dx_off in [0, 1]:
                yy = y0 + dy_off
                xx = x0 + dx_off
                wt = (1 - np.abs(wx - dx_off)) * (1 - np.abs(wy - dy_off))

                valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
                yy_v = yy[valid]
                xx_v = xx[valid]
                wt_v = wt[valid]

                np.add.at(weight, (yy_v, xx_v), wt_v)
                for c in range(3):
                    np.add.at(output[..., c], (yy_v, xx_v), pixels[..., c][valid] * wt_v)

        mask = weight > 0
        for c in range(3):
            output[..., c][mask] /= weight[mask]

        return output.astype(np.uint8), mask

    def interpolate_frame_splat(self, frame_before, frame_after, alpha=0.5):
        """Bidirectional DIS optical flow with forward splatting.
        Uses DIS (Dense Inverse Search) flow with bilateral smoothing,
        bidirectional splatting, inpainted holes, and light denoise.
        """
        gray_before = cv2.cvtColor(frame_before, cv2.COLOR_BGR2GRAY)
        gray_after = cv2.cvtColor(frame_after, cv2.COLOR_BGR2GRAY)

        # Compute DIS optical flow (better than Farneback)
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        dis.setFinestScale(0)
        dis.setGradientDescentIterations(50)
        dis.setPatchSize(12)
        dis.setPatchStride(4)

        flow_fwd = dis.calc(gray_before, gray_after, None)
        flow_bwd = dis.calc(gray_after, gray_before, None)

        # Bilateral filter on flow — smooth noise, preserve motion edges
        flow_fwd[..., 0] = cv2.bilateralFilter(flow_fwd[..., 0], 11, 50, 50)
        flow_fwd[..., 1] = cv2.bilateralFilter(flow_fwd[..., 1], 11, 50, 50)
        flow_bwd[..., 0] = cv2.bilateralFilter(flow_bwd[..., 0], 11, 50, 50)
        flow_bwd[..., 1] = cv2.bilateralFilter(flow_bwd[..., 1], 11, 50, 50)

        # Splat both directions
        splat_fwd, mask_fwd = self._forward_splat(frame_before, flow_fwd, alpha)
        splat_bwd, mask_bwd = self._forward_splat(frame_after, flow_bwd, 1 - alpha)

        # Composite: bidirectional blend with cross-dissolve fallback
        blend = cv2.addWeighted(frame_before, 1 - alpha, frame_after, alpha, 0)

        both = mask_fwd & mask_bwd
        fwd_only = mask_fwd & ~mask_bwd
        bwd_only = ~mask_fwd & mask_bwd
        neither = ~mask_fwd & ~mask_bwd

        result = np.zeros_like(frame_before, dtype=np.float64)
        result[both] = (splat_fwd[both].astype(np.float64) +
                         splat_bwd[both].astype(np.float64)) / 2
        result[fwd_only] = splat_fwd[fwd_only]
        result[bwd_only] = splat_bwd[bwd_only]
        result[neither] = blend[neither]
        result = result.astype(np.uint8)

        # Inpaint any remaining holes using surrounding pixels
        combined_mask = mask_fwd | mask_bwd
        if not combined_mask.all():
            hole_mask = (~combined_mask).astype(np.uint8) * 255
            result = cv2.inpaint(result, hole_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

        # Light denoise to clean splat artifacts
        result = cv2.fastNlMeansDenoisingColored(result, None, 3, 3, 7, 21)

        return result

    def _init_rife(self):
        """Lazy-load RIFE model on first use."""
        want_fp32 = self.full_precision_var.get()
        # Reload if precision changed
        if hasattr(self, '_rife_model') and hasattr(self, '_rife_fp32') and self._rife_fp32 != want_fp32:
            del self._rife_model
        if not hasattr(self, '_rife_model'):
            from ccvfi import AutoModel, ConfigType, VFIBaseModel
            self._rife_model: VFIBaseModel = AutoModel.from_pretrained(
                pretrained_model_name=ConfigType.RIFE_IFNet_v426_heavy,
            )
            if want_fp32:
                self._rife_model.fp16 = False
                self._rife_model.model = self._rife_model.model.float()
            self._rife_fp32 = want_fp32
            mode = "fp32 (full precision)" if want_fp32 else "fp16 (half precision)"
            print(f"RIFE model loaded (IFNet v4.26 heavy) — {mode}")

    def interpolate_frame_rife(self, frame_before, frame_after, alpha=0.5):
        """Generate an in-between frame using RIFE neural network.
        RIFE directly outputs the interpolated frame — no optical flow
        splatting needed. Much higher quality than traditional methods.
        """
        import torch
        self._init_rife()

        device = self._rife_model.device

        dtype = torch.float16 if self._rife_model.fp16 else torch.float32
        img0 = torch.from_numpy(frame_before.copy()).permute(2, 0, 1).unsqueeze(0).to(dtype).to(device) / 255.0
        img1 = torch.from_numpy(frame_after.copy()).permute(2, 0, 1).unsqueeze(0).to(dtype).to(device) / 255.0

        # Stack as [1, 2, C, H, W] — ccvfi expects both frames concatenated
        imgs = torch.cat([img0, img1], dim=0).unsqueeze(0)  # [1, 2, C, H, W]

        with torch.no_grad():
            output = self._rife_model.inference(imgs, timestep=alpha, scale=1.0)

        result = (output.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return result

    def _burn_label(self, frame, text):
        """Burn a debug label into the lower-left corner of a frame (in-place)."""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.5, min(w, h) / 800)  # scale with resolution
        thickness = max(1, int(scale * 2))
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        # Background rectangle
        margin = 8
        x = margin
        y = h - margin
        cv2.rectangle(frame, (x - 4, y - th - 8), (x + tw + 4, y + baseline + 4),
                       (0, 0, 0), cv2.FILLED)
        # White text
        cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness,
                     cv2.LINE_AA)

    def fix_video(self):
        fix_mode = self.fix_mode_var.get()
        dip_entries = getattr(self, 'dip_entries', set())

        # Build insert plan (spikes) and replace plan (dips) from the editable table
        insert_plan = {}
        replace_plan = {}
        for frame_num, var in self.spike_entries.items():
            count = var.get()
            if count <= 0:
                continue
            is_dip = frame_num in dip_entries

            if is_dip and fix_mode in ("Dips Only", "Spikes + Dips"):
                replace_plan[frame_num] = True
            elif not is_dip and fix_mode in ("Spikes Only", "Spikes + Dips"):
                insert_plan[frame_num] = count

        if not insert_plan and not replace_plan:
            messagebox.showinfo("Nothing to do", "No fixes to apply in the selected mode.")
            return

        out_fmt = self.output_format_var.get()

        # If H.264 selected and FFmpeg is available, suggest ProRes
        if "H.264" in out_fmt and shutil.which("ffmpeg"):
            switch = messagebox.askyesno(
                "Output Format",
                "H.264 is lossy even at high bitrate.\n"
                "ProRes HQ is lossless and better for editing.\n\n"
                "Switch to ProRes HQ?",
                icon="question"
            )
            if switch:
                out_fmt = "ProRes HQ (.mov)"
                self.output_format_var.set(out_fmt)

        if "ProRes" in out_fmt:
            ext = ".mov"
            filetypes = [("QuickTime MOV", "*.mov")]
        else:
            ext = ".mp4"
            filetypes = [("MP4 files", "*.mp4")]

        output_path = filedialog.asksaveasfilename(
            title="Save fixed video",
            defaultextension=ext,
            filetypes=filetypes,
            initialfile=self.video_path.stem + "_fixed" + ext
        )
        if not output_path:
            return

        self.set_status("Building fixed video with interpolated frames...")
        self.fix_btn.configure(state=tk.DISABLED)

        def do_fix():
            try:
                fill_mode = self.fill_mode_var.get()

                use_prores = "ProRes" in out_fmt
                has_ffmpeg = shutil.which("ffmpeg")

                class FFmpegWriter:
                    def __init__(self, proc):
                        self.proc = proc
                    def write(self, frame):
                        self.proc.stdin.write(frame.tobytes())
                    def release(self):
                        self.proc.stdin.close()
                        self.proc.wait()

                if use_prores and has_ffmpeg:
                    # ProRes HQ via FFmpeg
                    proc = subprocess.Popen([
                        "ffmpeg", "-y",
                        "-f", "rawvideo",
                        "-pix_fmt", "bgr24",
                        "-s", f"{self.width}x{self.height}",
                        "-r", str(self.fps),
                        "-i", "-",
                        "-c:v", "prores_ks",
                        "-profile:v", "3",
                        "-pix_fmt", "yuv422p10le",
                        "-vendor", "apl0",
                        output_path
                    ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                    out = FFmpegWriter(proc)

                elif has_ffmpeg:
                    # H.264 via FFmpeg at ~25 Mbps (high quality)
                    proc = subprocess.Popen([
                        "ffmpeg", "-y",
                        "-f", "rawvideo",
                        "-pix_fmt", "bgr24",
                        "-s", f"{self.width}x{self.height}",
                        "-r", str(self.fps),
                        "-i", "-",
                        "-c:v", "libx264",
                        "-preset", "slow",
                        "-b:v", "25M",
                        "-pix_fmt", "yuv420p",
                        output_path
                    ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                    out = FFmpegWriter(proc)

                else:
                    # Fallback: H.264 via OpenCV (lower quality)
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    out = cv2.VideoWriter(output_path, fourcc, self.fps,
                                           (self.width, self.height))
                    if not out.isOpened():
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(output_path, fourcc, self.fps,
                                               (self.width, self.height))

                inserted = 0
                removed = 0
                total_fixes = len(insert_plan) + len(replace_plan)
                done = 0
                last_written = None  # track last frame we actually wrote

                for i, (frame_idx, frame) in enumerate(self.frames_data):
                    # --- Dip: skip (delete) the duplicate frame entirely ---
                    if frame_idx in replace_plan:
                        removed += 1
                        done += 1
                        self.root.after(0, lambda d=done, tf=total_fixes, fi=frame_idx: self.set_status(
                            f"Fix {d}/{tf} — removed dip frame F{fi}"
                        ))
                        continue  # don't write this frame at all

                    # --- Spike: insert interpolated frames before writing the spike ---
                    if frame_idx in insert_plan:
                        insert_count = insert_plan[frame_idx]
                        # Use last_written instead of frames_data[i-1]
                        # This skips over any dips that were removed
                        prev_frame = last_written

                        for k in range(insert_count):
                            alpha = (k + 1) / (insert_count + 1)

                            if fill_mode == "Black":
                                fill = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                                out.write(fill)

                            elif fill_mode == "White":
                                fill = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
                                out.write(fill)

                            elif fill_mode == "Blend (debug)":
                                if prev_frame is not None:
                                    blended = cv2.addWeighted(prev_frame, 1 - alpha, frame, alpha, 0)
                                    out.write(blended)
                                else:
                                    out.write(frame)

                            elif fill_mode == "RIFE (AI)":
                                if prev_frame is not None:
                                    interp = self.interpolate_frame_rife(prev_frame, frame, alpha)
                                    out.write(interp)
                                else:
                                    out.write(frame)

                            elif fill_mode == "RIFE + Labels":
                                if prev_frame is not None:
                                    interp = np.ascontiguousarray(self.interpolate_frame_rife(prev_frame, frame, alpha))
                                else:
                                    interp = frame.copy()
                                label = f"INSERT {k+1}/{insert_count} @ F{frame_idx}"
                                self._burn_label(interp, label)
                                out.write(interp)

                            inserted += 1

                        done += 1
                        self.root.after(0, lambda d=done, ic=insert_count, tf=total_fixes: self.set_status(
                            f"Fix {d}/{tf} — inserted {ic} frame(s)..."
                        ))

                    # Write the original frame (spike frames stay, normal frames stay)
                    out.write(frame)
                    last_written = frame

                out.release()

                codec_name = "ProRes HQ" if use_prores else "H.264"
                self.root.after(0, lambda: (
                    self.log_msg(f"\nDone! Inserted {inserted} frames, removed {removed} dip frames.\n", "good"),
                    self.log_msg(f"Fill: {fill_mode} | Codec: {codec_name}\n", "good"),
                    self._log_clickable_path(output_path),
                    self.set_status(f"Saved: {output_path}"),
                    self.fix_btn.configure(state=tk.NORMAL)
                ))
            except Exception as e:
                self.root.after(0, lambda: (
                    messagebox.showerror("Error", str(e)),
                    self.fix_btn.configure(state=tk.NORMAL)
                ))

        threading.Thread(target=do_fix, daemon=True).start()

    def stop_recursive(self):
        """Signal the recursive loop to stop after current pass."""
        self._stop_recursive = True
        self.set_status("Stopping after current pass...")

    def recursive_fix(self):
        """Recursive fix: analyze output, bump insert counts, re-render from originals."""
        insert_plan = {}
        for frame_num, var in self.spike_entries.items():
            count = var.get()
            if count > 0:
                insert_plan[frame_num] = count

        if not insert_plan:
            messagebox.showinfo("Nothing to do", "All insert counts are 0.")
            return

        out_fmt = self.output_format_var.get()
        ext = ".mov" if "ProRes" in out_fmt else ".mp4"

        # Create a subfolder for recursive output
        rec_folder = self.video_path.parent / (self.video_path.stem + "_recursive")

        # Confirm folder with user
        folder_str = filedialog.askdirectory(
            title="Choose folder for recursive output",
            initialdir=str(rec_folder.parent),
        )
        if not folder_str:
            return
        rec_folder = Path(folder_str) / (self.video_path.stem + "_recursive")
        rec_folder.mkdir(exist_ok=True)

        self.log_msg(f"\nRecursive output folder: {rec_folder}\n", "header")

        self._stop_recursive = False
        self._rec_folder = rec_folder
        self._rec_ext = ext
        self.fix_btn.configure(state=tk.DISABLED)
        self.recursive_btn.configure(state=tk.DISABLED)
        self.analyze_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)

        # Clear mini graphs
        for w in self.mini_graph_frame.winfo_children():
            w.destroy()

        fill_mode = self.fill_mode_var.get()
        # Force RIFE for recursive (labels don't make sense here)
        if fill_mode in ("Black", "White", "Blend (debug)", "RIFE + Labels"):
            fill_mode = "RIFE (AI)"

        # Final output path
        output_path = str(rec_folder / (self.video_path.stem + "_final" + ext))

        threading.Thread(target=self._do_recursive,
                         args=(insert_plan, output_path, fill_mode, out_fmt),
                         daemon=True).start()

    def _do_recursive(self, insert_plan, output_path, fill_mode, out_fmt):
        max_passes = 4
        base_threshold = self.threshold_var.get()
        intensity = self.intensity_var.get()
        # Aggressive drop: intensity 1 = 0.15/pass, intensity 5 = 0.75/pass, intensity 10 = 1.50/pass
        drop_per_pass = intensity * 0.15
        pass_colors = ["#ff6b6b", "#ffaa44", "#44dd44", "#44aaff"]
        rec_folder = self._rec_folder
        ext = self._rec_ext
        stem = self.video_path.stem
        consecutive_clean = 0

        try:
            for pass_num in range(1, max_passes + 1):
                if self._stop_recursive:
                    self.root.after(0, lambda p=pass_num: self.log_msg(
                        f"\nStopped after pass {p-1}.\n", "header"))
                    break

                # Tighten threshold each pass — start tightening from pass 1
                # Pass 1 already drops by one step so the tool actually catches subtler spikes
                threshold = max(1.05, base_threshold - pass_num * drop_per_pass)

                self.root.after(0, lambda p=pass_num, t=threshold: self.set_status(
                    f"Recursive pass {p}/{max_passes} (threshold {t:.2f}) — building frames..."))

                # Build output frame sequence from ORIGINAL frames
                output_frames, output_map = self._build_fixed_frames(insert_plan, fill_mode)

                # Save this pass to disk
                pass_path = str(rec_folder / f"{stem}_pass{pass_num}{ext}")
                self.root.after(0, lambda p=pass_num, pp=pass_path: self.set_status(
                    f"Recursive pass {p}/{max_passes} — saving pass..."))
                self._render_frames_to_disk(output_frames, pass_path, out_fmt)
                self.root.after(0, lambda p=pass_num, pp=pass_path:
                    self.log_msg(f"  Saved: {pp}\n", "good"))

                self.root.after(0, lambda p=pass_num: self.set_status(
                    f"Recursive pass {p}/{max_passes} — analyzing output..."))

                # Analyze the output for remaining spikes
                mags = self._analyze_frame_list(output_frames)
                spikes = self.detect_spikes(mags, threshold)

                # Draw mini graph for this pass
                color = pass_colors[pass_num - 1] if pass_num <= len(pass_colors) else "#aaaaaa"
                self.root.after(0, lambda p=pass_num, m=mags, s=spikes, c=color:
                    self._draw_mini_graph(p, m, s, c))

                spike_count = len(spikes)
                total_inserts = sum(insert_plan.values())
                self.root.after(0, lambda p=pass_num, sc=spike_count, ti=total_inserts, t=threshold:
                    self.log_msg(f"  Pass {p} (thresh {t:.2f}): {sc} spikes, {ti} total inserts\n",
                                 "spike" if sc > 0 else "good"))

                if not spikes:
                    consecutive_clean += 1
                    # Only stop early if we've had 2 clean passes in a row
                    # (meaning tighter threshold didn't reveal new spikes either)
                    if consecutive_clean >= 2:
                        self.root.after(0, lambda p=pass_num: self.log_msg(
                            f"\nClean for 2 consecutive passes — done!\n", "good"))
                        break
                    else:
                        self.root.after(0, lambda p=pass_num, t=threshold: self.log_msg(
                            f"  Pass {p} clean at {t:.2f} — continuing with tighter threshold...\n", "header"))
                        continue
                else:
                    consecutive_clean = 0

                if self._stop_recursive:
                    break

                # Map remaining spikes back to original positions and bump counts
                for spike in spikes:
                    output_idx = spike['frame']
                    orig_frame = self._map_to_original(output_idx, output_map)
                    if orig_frame is not None:
                        if orig_frame in insert_plan:
                            insert_plan[orig_frame] += 1
                        else:
                            insert_plan[orig_frame] = 1

            # Update the spike table to reflect final counts
            self.root.after(0, lambda: self._rebuild_spike_table(insert_plan))

            total = sum(insert_plan.values())
            self.root.after(0, lambda: (
                self.log_msg(f"\nRecursive fix complete! {total} frames inserted.\n", "good"),
                self.log_msg(f"All passes saved to: {rec_folder}\n", "good"),
                self.set_status(f"Done — {rec_folder}")
            ))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, lambda: (
                self.fix_btn.configure(state=tk.NORMAL),
                self.recursive_btn.configure(state=tk.NORMAL),
                self.analyze_btn.configure(state=tk.NORMAL),
                self.stop_btn.configure(state=tk.DISABLED)
            ))

    def _build_fixed_frames(self, insert_plan, fill_mode):
        """Build the output frame sequence in memory from original frames.
        Returns (output_frames, output_map) where output_map[i] = original_frame_idx or None.
        """
        output_frames = []
        output_map = []  # original frame idx for each output frame, None = inserted

        for i, (frame_idx, frame) in enumerate(self.frames_data):
            if frame_idx in insert_plan:
                insert_count = insert_plan[frame_idx]
                prev_frame = self.frames_data[i - 1][1] if i > 0 else None

                for k in range(insert_count):
                    alpha = (k + 1) / (insert_count + 1)
                    if prev_frame is not None:
                        if "RIFE" in fill_mode:
                            interp = self.interpolate_frame_rife(prev_frame, frame, alpha)
                        elif fill_mode == "InterpolateNew":
                            interp = self.interpolate_frame_splat(prev_frame, frame, alpha)
                        else:
                            interp = cv2.addWeighted(prev_frame, 1 - alpha, frame, alpha, 0)
                    else:
                        interp = frame.copy()
                    output_frames.append(interp)
                    output_map.append(None)  # inserted frame

            output_frames.append(frame)
            output_map.append(frame_idx)  # original frame

        return output_frames, output_map

    def _analyze_frame_list(self, frames):
        """Run optical flow magnitude analysis on a list of frames (in memory)."""
        if len(frames) < 2:
            return []

        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        magnitudes = []

        for i in range(1, len(frames)):
            gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            avg_mag = float(np.mean(mag))
            magnitudes.append((i, avg_mag))
            prev_gray = gray

            # Update progress
            if i % 10 == 0:
                pct = (i / len(frames)) * 100
                self.root.after(0, lambda p=pct: self.progress_var.set(p))

        return magnitudes

    def _map_to_original(self, output_idx, output_map):
        """Map an output frame index back to the original frame that needs more inserts.
        A spike at output_idx means the transition INTO that frame is too abrupt.
        Find the next original frame at or after output_idx — that's the one needing more inserts.
        """
        # Look at the spike frame and forward to find the original frame
        for j in range(output_idx, len(output_map)):
            if output_map[j] is not None:
                return output_map[j]
        # Look backward as fallback
        for j in range(output_idx - 1, -1, -1):
            if output_map[j] is not None:
                return output_map[j]
        return None

    def _draw_mini_graph(self, pass_num, magnitudes, spikes, color):
        """Draw a small motion graph for a recursive pass."""
        row = ttk.Frame(self.mini_graph_frame, style="Dark.TFrame")
        row.pack(fill=tk.X, pady=1)

        label = ttk.Label(row, text=f"Pass {pass_num}:", style="Dark.TLabel", width=7,
                           font=("Consolas", 8))
        label.pack(side=tk.LEFT)

        spike_text = f"{len(spikes)} spikes" if spikes else "clean!"
        info = ttk.Label(row, text=spike_text, style="Dark.TLabel", width=10,
                          font=("Consolas", 8),
                          foreground="#ff6b6b" if spikes else "#6bff6b")
        info.pack(side=tk.LEFT)

        mini = tk.Canvas(row, bg="#1a1a1a", height=30, highlightthickness=0)
        mini.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        if not magnitudes:
            return

        mini.update_idletasks()
        w = mini.winfo_width()
        if w < 10:
            w = 500
        h = 30

        mags = [m[1] for m in magnitudes]
        max_mag = max(mags) if mags else 1
        n = len(mags)
        spike_frames = set(s['frame'] for s in spikes)

        for i, mag in enumerate(mags):
            x = (i / n) * w
            bar_h = (mag / max_mag) * (h - 4) if max_mag > 0 else 0
            y = h - bar_h
            frame_num = magnitudes[i][0]
            c = "#ff4444" if frame_num in spike_frames else color
            mini.create_line(x, h, x, y, fill=c, width=1)

    def _render_frames_to_disk(self, frames, output_path, out_fmt):
        """Write a list of frames directly to disk (no re-interpolation)."""
        use_prores = "ProRes" in out_fmt

        if use_prores:
            ffmpeg_proc = subprocess.Popen([
                "ffmpeg", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{self.width}x{self.height}",
                "-r", str(self.fps), "-i", "-",
                "-c:v", "prores_ks", "-profile:v", "3",
                "-pix_fmt", "yuv422p10le", "-vendor", "apl0",
                output_path
            ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)

            for frame in frames:
                ffmpeg_proc.stdin.write(frame.tobytes())
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()
        else:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, self.fps,
                                   (self.width, self.height))
            if not out.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, self.fps,
                                       (self.width, self.height))
            for frame in frames:
                out.write(frame)
            out.release()

    def _render_to_disk(self, insert_plan, output_path, fill_mode, out_fmt):
        """Final render from original frames with the given insert plan."""
        use_prores = "ProRes" in out_fmt

        if use_prores:
            ffmpeg_proc = subprocess.Popen([
                "ffmpeg", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{self.width}x{self.height}",
                "-r", str(self.fps), "-i", "-",
                "-c:v", "prores_ks", "-profile:v", "3",
                "-pix_fmt", "yuv422p10le", "-vendor", "apl0",
                output_path
            ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)

            class FFmpegWriter:
                def __init__(self, proc):
                    self.proc = proc
                def write(self, frame):
                    self.proc.stdin.write(frame.tobytes())
                def release(self):
                    self.proc.stdin.close()
                    self.proc.wait()

            out = FFmpegWriter(ffmpeg_proc)
        else:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, self.fps,
                                   (self.width, self.height))
            if not out.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, self.fps,
                                       (self.width, self.height))

        total = len(self.frames_data)
        for i, (frame_idx, frame) in enumerate(self.frames_data):
            if frame_idx in insert_plan:
                insert_count = insert_plan[frame_idx]
                prev_frame = self.frames_data[i - 1][1] if i > 0 else None

                for k in range(insert_count):
                    alpha = (k + 1) / (insert_count + 1)
                    if prev_frame is not None:
                        if "RIFE" in fill_mode:
                            interp = self.interpolate_frame_rife(prev_frame, frame, alpha)
                        elif fill_mode == "InterpolateNew":
                            interp = self.interpolate_frame_splat(prev_frame, frame, alpha)
                        else:
                            interp = cv2.addWeighted(prev_frame, 1 - alpha, frame, alpha, 0)
                    else:
                        interp = frame.copy()
                    out.write(interp)

            out.write(frame)

            pct = (i / total) * 100
            self.root.after(0, lambda p=pct: self.progress_var.set(p))

        out.release()

    def export_report(self):
        output_path = filedialog.asksaveasfilename(
            title="Save analysis report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            initialfile=self.video_path.stem + "_analysis.txt"
        )
        if not output_path:
            return

        with open(output_path, "w") as f:
            f.write(f"Frame Detective Analysis Report\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"File: {self.video_path}\n")
            f.write(f"Resolution: {self.width}x{self.height}\n")
            f.write(f"FPS: {self.fps:.2f}\n")
            f.write(f"Total frames: {len(self.magnitudes)}\n")
            f.write(f"Threshold: {self.threshold_var.get()}\n")
            f.write(f"Spikes found: {len(self.spikes)}\n\n")

            if self.spikes:
                f.write("Spike Details:\n")
                f.write(f"{'-' * 60}\n")
                for s in self.spikes:
                    f.write(f"  Frame {s['frame']:4d}  |  motion: {s['magnitude']:.2f}  |  "
                            f"local avg: {s['local_median']:.2f}  |  {s['ratio']:.1f}x\n")

            f.write(f"\nAll Frame Magnitudes:\n")
            f.write(f"{'-' * 60}\n")
            spike_set = set(s['frame'] for s in self.spikes)
            for frame_idx, mag in self.magnitudes:
                marker = " <<< SPIKE" if frame_idx in spike_set else ""
                f.write(f"  {frame_idx:4d}  |  {mag:.4f}{marker}\n")

        self.log_msg(f"\nReport saved: {output_path}\n", "good")
        self.set_status(f"Report saved.")


def main():
    root = tk.Tk()
    app = FrameDetectiveApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
