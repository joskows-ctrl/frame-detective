"""
Frame Detective — Detect missing frames in AI-generated video
Uses optical flow magnitude to find abnormal jumps between frames,
then inserts black frames where gaps are detected to smooth playback.

Usage:
  python detect_missing_frames.py input.mp4
  python detect_missing_frames.py input.mp4 --threshold 1.8 --output fixed.mp4
  python detect_missing_frames.py input.mp4 --analyze-only
"""

import cv2
import numpy as np
import argparse
import sys
from pathlib import Path


def compute_flow_magnitudes(video_path):
    """Compute optical flow magnitude between consecutive frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps:.2f}fps, {total} frames")
    print(f"Analyzing optical flow...")

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        sys.exit(1)

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    magnitudes = []
    frames_data = [(0, prev_frame.copy())]  # (frame_idx, frame)

    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute dense optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # Magnitude of flow vectors
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        avg_mag = np.mean(mag)
        magnitudes.append((frame_idx, avg_mag))
        frames_data.append((frame_idx, frame.copy()))

        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx}/{total} — avg motion: {avg_mag:.2f}")

        prev_gray = gray
        frame_idx += 1

    cap.release()
    return magnitudes, frames_data, fps, width, height


def detect_spikes(magnitudes, threshold_multiplier=2.0):
    """Find frames where motion jumps abnormally (missing frame indicators)."""
    if len(magnitudes) < 5:
        return []

    mags = np.array([m[1] for m in magnitudes])

    # Use rolling median to handle varying motion (camera moves, etc.)
    window = min(15, len(mags) // 2)
    if window < 3:
        window = 3

    spikes = []
    for i in range(len(mags)):
        # Local window around this frame
        start = max(0, i - window)
        end = min(len(mags), i + window + 1)
        local = np.concatenate([mags[start:i], mags[i+1:end]])  # exclude self

        if len(local) == 0:
            continue

        local_median = np.median(local)
        local_std = np.std(local)

        # Spike if magnitude is significantly above local context
        if local_median > 0:
            ratio = mags[i] / local_median
            if ratio > threshold_multiplier and mags[i] > local_median + local_std:
                spikes.append({
                    'frame': magnitudes[i][0],
                    'magnitude': mags[i],
                    'local_median': local_median,
                    'ratio': ratio
                })

    return spikes


def build_fixed_video(frames_data, spikes, fps, width, height, output_path, black_frames=1):
    """Rebuild video with black frames inserted at spike points."""
    spike_frames = set(s['frame'] for s in spikes)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    black = np.zeros((height, width, 3), dtype=np.uint8)
    inserted = 0

    for frame_idx, frame in frames_data:
        # Insert black frame(s) BEFORE the spike frame
        if frame_idx in spike_frames:
            for _ in range(black_frames):
                out.write(black)
                inserted += 1
        out.write(frame)

    out.release()
    return inserted


def print_analysis(magnitudes, spikes):
    """Print a visual chart of motion magnitudes with spikes marked."""
    mags = [m[1] for m in magnitudes]
    max_mag = max(mags) if mags else 1
    chart_width = 60

    print("\n" + "=" * 80)
    print("MOTION MAGNITUDE CHART (spikes marked with >>>)")
    print("=" * 80)

    spike_frames = set(s['frame'] for s in spikes)

    for frame_idx, mag in magnitudes:
        bar_len = int((mag / max_mag) * chart_width)
        bar = "█" * bar_len
        marker = " >>> SPIKE" if frame_idx in spike_frames else ""
        if frame_idx % 3 == 0 or frame_idx in spike_frames:  # show every 3rd + spikes
            print(f"  {frame_idx:4d} |{bar}{marker}")

    print("=" * 80)
    print(f"\nTotal frames analyzed: {len(magnitudes)}")
    print(f"Spikes detected: {len(spikes)}")

    if spikes:
        print(f"\nSpike details:")
        for s in spikes:
            print(f"  Frame {s['frame']:4d} — motion: {s['magnitude']:.2f} "
                  f"(local avg: {s['local_median']:.2f}, ratio: {s['ratio']:.1f}x)")


def main():
    parser = argparse.ArgumentParser(description="Detect missing frames in AI-generated video")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("--output", "-o", help="Output video path (default: input_fixed.mp4)")
    parser.add_argument("--threshold", "-t", type=float, default=2.0,
                        help="Spike detection threshold multiplier (default: 2.0, lower = more sensitive)")
    parser.add_argument("--black-frames", "-b", type=int, default=1,
                        help="Number of black frames to insert at each spike (default: 1)")
    parser.add_argument("--analyze-only", "-a", action="store_true",
                        help="Only analyze and show chart, don't create output video")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    # Analyze
    magnitudes, frames_data, fps, width, height = compute_flow_magnitudes(input_path)

    # Detect spikes
    spikes = detect_spikes(magnitudes, args.threshold)

    # Print results
    print_analysis(magnitudes, spikes)

    if args.analyze_only:
        return

    if not spikes:
        print("\nNo spikes detected — video looks clean!")
        return

    # Build fixed video
    output_path = args.output or input_path.stem + "_fixed.mp4"
    output_path = Path(output_path)

    print(f"\nInserting {args.black_frames} black frame(s) at {len(spikes)} spike locations...")
    inserted = build_fixed_video(frames_data, spikes, fps, width, height, output_path, args.black_frames)
    print(f"Done! Inserted {inserted} black frames.")
    print(f"Output: {output_path}")
    print(f"\nTip: Compare original vs fixed side-by-side to verify the spikes were real stutters.")


if __name__ == "__main__":
    main()
