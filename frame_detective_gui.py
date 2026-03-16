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


class FrameDetectiveApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Frame Detective")
        self.root.geometry("820x700")
        self.root.configure(bg="#1a1a1a")
        self.root.resizable(True, True)

        self.video_path = None
        self.magnitudes = []
        self.frames_data = []
        self.spikes = []
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

        # Title
        ttk.Label(main, text="Frame Detective", style="Title.TLabel").pack(anchor="w")
        ttk.Label(main, text="Detect missing frames in AI-generated video",
                  style="Info.TLabel").pack(anchor="w", pady=(0, 16))

        # File selection row
        file_frame = ttk.Frame(main, style="Dark.TFrame")
        file_frame.pack(fill=tk.X, pady=(0, 12))

        self.file_label = ttk.Label(file_frame, text="No file selected",
                                     style="Dark.TLabel", width=60)
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        browse_btn = ttk.Button(file_frame, text="Browse MP4...", command=self.browse_file)
        browse_btn.pack(side=tk.RIGHT, padx=(8, 0))

        # Settings row
        settings_frame = ttk.LabelFrame(main, text="Settings", style="Dark.TLabelframe",
                                         padding=12)
        settings_frame.pack(fill=tk.X, pady=(0, 12))

        settings_inner = ttk.Frame(settings_frame, style="Dark.TFrame")
        settings_inner.pack(fill=tk.X)

        # Threshold
        ttk.Label(settings_inner, text="Threshold:", style="Dark.TLabel").pack(side=tk.LEFT)
        self.threshold_var = tk.DoubleVar(value=2.0)
        thresh_spin = ttk.Spinbox(settings_inner, from_=1.2, to=5.0, increment=0.1,
                                   textvariable=self.threshold_var, width=6)
        thresh_spin.pack(side=tk.LEFT, padx=(4, 16))

        ttk.Label(settings_inner, text="(lower = more sensitive)", style="Info.TLabel").pack(side=tk.LEFT)

        # Second settings row
        settings_inner2 = ttk.Frame(settings_frame, style="Dark.TFrame")
        settings_inner2.pack(fill=tk.X, pady=(8, 0))

        # Auto vs manual frame count
        self.auto_frames_var = tk.BooleanVar(value=True)
        auto_cb = ttk.Checkbutton(settings_inner2, text="Auto-detect count",
                                   variable=self.auto_frames_var, command=self.toggle_manual)
        auto_cb.pack(side=tk.LEFT)

        ttk.Label(settings_inner2, text="Manual:", style="Dark.TLabel").pack(side=tk.LEFT, padx=(16, 0))
        self.insert_count_var = tk.IntVar(value=1)
        self.manual_spin = ttk.Spinbox(settings_inner2, from_=1, to=5, increment=1,
                               textvariable=self.insert_count_var, width=4, state=tk.DISABLED)
        self.manual_spin.pack(side=tk.LEFT, padx=(4, 0))

        # Fill mode: Black / White / Blend (debug) / Interpolate (RAFT)
        ttk.Label(settings_inner2, text="Fill mode:", style="Dark.TLabel").pack(side=tk.LEFT, padx=(24, 0))
        self.fill_mode_var = tk.StringVar(value="Black")
        self.fill_mode_combo = ttk.Combobox(settings_inner2, textvariable=self.fill_mode_var,
                                             values=["Black", "White", "Blend (debug)", "Interpolate (RAFT)"],
                                             state="readonly", width=18)
        self.fill_mode_combo.pack(side=tk.LEFT, padx=(4, 0))

        # Buttons row
        btn_frame = ttk.Frame(main, style="Dark.TFrame")
        btn_frame.pack(fill=tk.X, pady=(0, 12))

        self.analyze_btn = ttk.Button(btn_frame, text="Analyze", command=self.start_analyze,
                                       style="Accent.TButton")
        self.analyze_btn.pack(side=tk.LEFT)

        self.fix_btn = ttk.Button(btn_frame, text="Fix Video (Insert Frames)",
                                   command=self.fix_video, state=tk.DISABLED)
        self.fix_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.export_btn = ttk.Button(btn_frame, text="Export Report",
                                      command=self.export_report, state=tk.DISABLED)
        self.export_btn.pack(side=tk.LEFT, padx=(8, 0))

        # Progress
        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(main, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, pady=(0, 4))

        self.status_label = ttk.Label(main, text="Ready", style="Info.TLabel")
        self.status_label.pack(anchor="w", pady=(0, 8))

        # Results area — canvas for the chart + text log
        results_frame = ttk.LabelFrame(main, text="Results", style="Dark.TLabelframe",
                                        padding=8)
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Chart canvas
        self.canvas = tk.Canvas(results_frame, bg="#222222", height=200,
                                 highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        # Log text
        self.log = tk.Text(results_frame, bg="#222222", fg="#cccccc",
                           font=("Consolas", 9), height=8, wrap=tk.WORD,
                           insertbackground="#cccccc", relief=tk.FLAT)
        self.log.pack(fill=tk.BOTH, expand=True)

        # Tag for spike highlights
        self.log.tag_configure("spike", foreground="#ff6b6b", font=("Consolas", 9, "bold"))
        self.log.tag_configure("good", foreground="#6bff6b")
        self.log.tag_configure("header", foreground="#ffffff", font=("Consolas", 10, "bold"))

        # Enable drag & drop via simple protocol
        self.setup_drop_zone()

    def toggle_manual(self):
        if self.auto_frames_var.get():
            self.manual_spin.configure(state=tk.DISABLED)
        else:
            self.manual_spin.configure(state=tk.NORMAL)

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
        if path.lower().endswith('.mp4'):
            self.video_path = Path(path)
            self.file_label.configure(text=str(self.video_path))
        else:
            messagebox.showwarning("Invalid file", "Please drop an MP4 file.")

    def browse_file(self):
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("MP4 files", "*.mp4"), ("All video", "*.mp4 *.mov *.avi *.mkv *.webm"), ("All files", "*.*")]
        )
        if path:
            self.video_path = Path(path)
            self.file_label.configure(text=str(self.video_path))
            self.log_msg(f"Loaded: {self.video_path.name}\n")

    def log_msg(self, msg, tag=None):
        self.log.insert(tk.END, msg, tag)
        self.log.see(tk.END)

    def set_status(self, msg):
        self.status_label.configure(text=msg)

    def start_analyze(self):
        if not self.video_path or not self.video_path.exists():
            messagebox.showwarning("No file", "Please select an MP4 file first.")
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

            # Detect spikes
            threshold = self.threshold_var.get()
            self.spikes = self.detect_spikes(self.magnitudes, threshold)

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
                    # Estimate missing frames from ratio
                    # 2x = 1 missing, 3x = 2 missing, etc.
                    est_missing = max(1, round(ratio) - 1)
                    spikes.append({
                        'frame': magnitudes[i][0],
                        'magnitude': float(mags[i]),
                        'local_median': local_median,
                        'ratio': float(ratio),
                        'est_missing': est_missing
                    })
        return spikes

    def show_results(self):
        self.progress_var.set(100)
        self.draw_chart()

        spike_frames = set(s['frame'] for s in self.spikes)

        self.log_msg(f"\nAnalysis complete: {len(self.magnitudes)} frames\n", "header")

        if self.spikes:
            self.log_msg(f"Found {len(self.spikes)} suspected missing frames:\n\n", "spike")
            for s in self.spikes:
                self.log_msg(
                    f"  Frame {s['frame']:4d}  |  motion: {s['magnitude']:.2f}  |  "
                    f"local avg: {s['local_median']:.2f}  |  {s['ratio']:.1f}x spike  |  "
                    f"~{s['est_missing']} missing\n",
                    "spike"
                )
            self.fix_btn.configure(state=tk.NORMAL)
            self.export_btn.configure(state=tk.NORMAL)
            self.set_status(f"Found {len(self.spikes)} spikes. Ready to fix.")
        else:
            self.log_msg("No spikes detected — video looks clean!\n", "good")
            self.export_btn.configure(state=tk.NORMAL)
            self.set_status("No issues found.")

    def draw_chart(self):
        """Draw motion magnitude chart on canvas."""
        self.canvas.delete("all")
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

        mags = [m[1] for m in self.magnitudes]
        max_mag = max(mags) if mags else 1
        n = len(mags)
        spike_frames = set(s['frame'] for s in self.spikes)

        # Axes
        c.create_line(pad_left, pad_top, pad_left, ch - pad_bottom, fill="#555")
        c.create_line(pad_left, ch - pad_bottom, cw - pad_right, ch - pad_bottom, fill="#555")

        # Y-axis label
        c.create_text(pad_left - 5, pad_top, text=f"{max_mag:.1f}", anchor="ne",
                       fill="#888", font=("Consolas", 8))
        c.create_text(pad_left - 5, ch - pad_bottom, text="0", anchor="ne",
                       fill="#888", font=("Consolas", 8))

        # X-axis labels
        c.create_text(pad_left, ch - pad_bottom + 12, text="0", anchor="n",
                       fill="#888", font=("Consolas", 8))
        c.create_text(cw - pad_right, ch - pad_bottom + 12, text=str(n), anchor="n",
                       fill="#888", font=("Consolas", 8))

        # Draw bars / line
        if n > 0:
            bar_w = max(1, plot_w / n)

            for i, mag in enumerate(mags):
                x = pad_left + (i / n) * plot_w
                h = (mag / max_mag) * plot_h if max_mag > 0 else 0
                y = ch - pad_bottom - h
                frame_num = self.magnitudes[i][0]

                if frame_num in spike_frames:
                    color = "#ff4444"
                    # Draw spike marker
                    c.create_rectangle(x, y, x + max(bar_w, 2), ch - pad_bottom,
                                        fill="#ff4444", outline="")
                    c.create_text(x, y - 8, text=f"F{frame_num}", fill="#ff6b6b",
                                   font=("Consolas", 7), anchor="s")
                else:
                    color = "#4a9eff"
                    c.create_rectangle(x, y, x + max(bar_w, 2), ch - pad_bottom,
                                        fill="#3a7acc", outline="")

        # Title
        c.create_text(cw // 2, 8, text="Motion Magnitude per Frame",
                       fill="#aaa", font=("Segoe UI", 9), anchor="n")

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

    def fix_video(self):
        if not self.spikes:
            return

        output_path = filedialog.asksaveasfilename(
            title="Save fixed video",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4")],
            initialfile=self.video_path.stem + "_fixed.mp4"
        )
        if not output_path:
            return

        self.set_status("Building fixed video with interpolated frames...")
        self.fix_btn.configure(state=tk.DISABLED)

        def do_fix():
            try:
                # Build spike lookup: frame_idx -> spike info
                spike_lookup = {s['frame']: s for s in self.spikes}
                use_auto = self.auto_frames_var.get()
                manual_count = self.insert_count_var.get()
                fill_mode = self.fill_mode_var.get()

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, self.fps,
                                       (self.width, self.height))

                inserted = 0
                total_spikes = len(self.spikes)
                done = 0

                for i, (frame_idx, frame) in enumerate(self.frames_data):
                    if frame_idx in spike_lookup:
                        spike = spike_lookup[frame_idx]
                        insert_count = spike['est_missing'] if use_auto else manual_count

                        # Get previous frame (before the spike)
                        prev_frame = self.frames_data[i - 1][1] if i > 0 else None

                        for k in range(insert_count):
                            alpha = (k + 1) / (insert_count + 1)

                            if fill_mode == "Black":
                                fill = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                                out.write(fill)

                            elif fill_mode == "White":
                                fill = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
                                out.write(fill)

                            elif fill_mode == "Blend (debug)":
                                # Simple cross-dissolve so you can see what frames it's using
                                if prev_frame is not None:
                                    blended = cv2.addWeighted(prev_frame, 1 - alpha, frame, alpha, 0)
                                    out.write(blended)
                                else:
                                    out.write(frame)

                            elif fill_mode == "Interpolate (RAFT)":
                                # Neural optical flow interpolation
                                if prev_frame is not None:
                                    interp = self.interpolate_frame(prev_frame, frame, alpha)
                                    out.write(interp)
                                else:
                                    out.write(frame)

                            inserted += 1

                        done += 1
                        self.root.after(0, lambda d=done, ic=insert_count, fm=fill_mode: self.set_status(
                            f"Spike {d}/{total_spikes} — {fm}: inserted {ic} frame(s)..."
                        ))

                    # Always write the original frame (spike frame stays)
                    out.write(frame)

                out.release()

                self.root.after(0, lambda: (
                    self.log_msg(f"\nFixed! Inserted {inserted} frames ({fill_mode}) at {total_spikes} spike locations.\n", "good"),
                    self.log_msg(f"Saved to: {output_path}\n", "good"),
                    self.set_status(f"Saved: {output_path}"),
                    self.fix_btn.configure(state=tk.NORMAL)
                ))
            except Exception as e:
                self.root.after(0, lambda: (
                    messagebox.showerror("Error", str(e)),
                    self.fix_btn.configure(state=tk.NORMAL)
                ))

        threading.Thread(target=do_fix, daemon=True).start()

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
