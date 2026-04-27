"""
Sleep Detection Calibration Script
====================================
Records two sessions (awake and sleeping) directly from webcam
using OpenFace, then computes optimal thresholds automatically.

Requirements:
    - OpenFace installed at C:/OpenFace
    - engagement_env activated

Usage:
    python calibrate_sleep.py
"""

import os
import sys
import time
import subprocess
import argparse
import numpy as np
import pandas as pd
import cv2
from pathlib import Path

# ── OpenFace Configuration ────────────────────────────────────────────────────
OPENFACE_DIR    = r"C:\OpenFace"
OPENFACE_EXE    = os.path.join(OPENFACE_DIR, "FeatureExtraction.exe")
OUTPUT_BASE_DIR = os.path.join(OPENFACE_DIR, "calibration_output")

# ── Recording Configuration ───────────────────────────────────────────────────
AWAKE_DURATION_SEC   = 30  # seconds to record awake session
SLEEPING_DURATION_SEC = 30  # seconds to record sleeping session

# ── Correct OpenFace Landmark Indices ─────────────────────────────────────────
# Correct OpenFace landmark indices (verified empirically)
# Left eye:  outer(8), top-outer(1), top-center(2), top-inner(3),
#            inner(14), bot-inner(5), bot-center(6), bot-outer(7)
# Right eye: same pattern + 28 offset
LEFT_EYE_IDX  = [8, 2, 3, 14, 5, 6]   # outer, top-outer, top-inner, inner, bot-inner, bot-outer
RIGHT_EYE_IDX = [36, 30, 31, 42, 33, 34]

# ── EAR Computation ───────────────────────────────────────────────────────────

def compute_ear(x, y):
    """Standard EAR formula using 6 eye landmark points."""
    v1 = np.sqrt((x[1] - x[5])**2 + (y[1] - y[5])**2)
    v2 = np.sqrt((x[2] - x[4])**2 + (y[2] - y[4])**2)
    h  = np.sqrt((x[0] - x[3])**2 + (y[0] - y[3])**2)
    return (v1 + v2) / (2.0 * h) if h > 0 else 0.0


def compute_ear_for_row(row):
    """Compute average EAR from one OpenFace CSV row."""
    try:
        lx = [float(row[f' eye_lmk_x_{i}']) for i in LEFT_EYE_IDX]
        ly = [float(row[f' eye_lmk_y_{i}']) for i in LEFT_EYE_IDX]
        rx = [float(row[f' eye_lmk_x_{i}']) for i in RIGHT_EYE_IDX]
        ry = [float(row[f' eye_lmk_y_{i}']) for i in RIGHT_EYE_IDX]
        return (compute_ear(lx, ly) + compute_ear(rx, ry)) / 2.0
    except Exception:
        return None


# ── Recording ─────────────────────────────────────────────────────────────────

def show_countdown(message, seconds, color=(0, 255, 255)):
    """Show a countdown window before recording starts."""
    cap = cv2.VideoCapture(0)
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        remaining = seconds - int(time.time() - start)
        if remaining <= 0:
            break

        # Overlay message and countdown
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, message, (30, frame.shape[0]//2 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Starting in {remaining}s...",
                    (30, frame.shape[0]//2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(frame, "Press Q to skip countdown",
                    (30, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("Sleep Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def show_recording_progress(duration_sec, label, color):
    """
    Show a live webcam window with recording progress bar.
    OpenFace runs in background — this is just for user feedback.
    """
    cap = cv2.VideoCapture(0)
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed  = time.time() - start
        remaining = max(0, duration_sec - elapsed)
        progress  = min(elapsed / duration_sec, 1.0)

        if elapsed >= duration_sec:
            break

        # Progress bar
        bar_x, bar_y, bar_w, bar_h = 20, frame.shape[0] - 50, frame.shape[1] - 40, 20
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + int(bar_w * progress), bar_y + bar_h),
                      color, -1)

        # Labels
        cv2.putText(frame, f"RECORDING: {label}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"Time remaining: {remaining:.1f}s", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame,
                    "AWAKE: Look at screen naturally" if label == "AWAKE"
                    else "SLEEPING: Close your eyes, bow head",
                    (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Sleep Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"\nRecording stopped early at {elapsed:.1f}s")
            break

    cap.release()
    cv2.destroyAllWindows()


def record_with_openface(output_dir, filename_prefix, duration_sec, label, color):
    """
    Launch OpenFace FeatureExtraction as a subprocess while
    showing progress to the user via a separate webcam window.
    """
    os.makedirs(output_dir, exist_ok=True)

    # OpenFace command
    cmd = [
        OPENFACE_EXE,
        "-device", "0",
        "-out_dir", output_dir,
        "-of", filename_prefix,
        "-aus",           # Force AU extraction
        "-pose",          # Force pose extraction
        "-2Dfp",          # 2D facial landmarks
        "-3Dfp",          # 3D facial landmarks
    ]

    print(f"\n  Launching OpenFace for {label} recording...")
    print(f"  Output: {output_dir}\\{filename_prefix}.csv")

    # Start OpenFace in background
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=OPENFACE_DIR
    )

    # Show progress window to user
    # Small delay to let OpenFace initialize
    time.sleep(2.0)
    show_recording_progress(duration_sec, label, color)

    # Stop OpenFace
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()

    print(f"  Recording complete.")

    # Find the generated CSV
    csv_path = os.path.join(output_dir, f"{filename_prefix}.csv")
    if not os.path.exists(csv_path):
        # OpenFace sometimes appends extra characters — find it
        candidates = list(Path(output_dir).glob(f"{filename_prefix}*.csv"))
        if candidates:
            csv_path = str(candidates[0])
        else:
            print(f"  ERROR: CSV not found in {output_dir}")
            return None

    print(f"  CSV found: {csv_path}")
    return csv_path


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze_csv(csv_path, label):
    """Load CSV, compute EAR and AU45 stats, return summary dict."""
    print(f"\n  Reading: {csv_path}")
    df = pd.read_csv(csv_path)

    # Filter to confident frames only
    if ' confidence' in df.columns and ' success' in df.columns:
        before = len(df)
        df = df[(df[' confidence'] >= 0.5) & (df[' success'] == 1)]
        print(f"  Frames after confidence filter: {len(df)} / {before}")

    total_frames = len(df)
    if total_frames == 0:
        print("  ERROR: No confident frames found.")
        return None

    # Compute EAR per row
    ears = []
    for _, row in df.iterrows():
        ear = compute_ear_for_row(row)
        if ear is not None:
            ears.append(ear)
    ears = np.array(ears)

    # AU45
    au45 = df[' AU45_r'].values if ' AU45_r' in df.columns else np.zeros(total_frames)

    # Confidence
    conf = df[' confidence'].values if ' confidence' in df.columns else np.ones(total_frames)

    separator = "=" * 70
    print(f"\n{separator}")
    print(f" CALIBRATION RESULTS — {label.upper()}")
    print(separator)
    print(f"\n  Frames analyzed: {total_frames}")

    print(f"\n  [EAR Statistics]:")
    print(f"    Min:    {ears.min():.4f}")
    print(f"    Max:    {ears.max():.4f}")
    print(f"    Mean:   {ears.mean():.4f}")
    print(f"    Std:    {ears.std():.4f}")
    print(f"    Q1:     {np.percentile(ears, 25):.4f}")
    print(f"    Median: {np.percentile(ears, 50):.4f}")
    print(f"    Q3:     {np.percentile(ears, 75):.4f}")

    print(f"\n  [AU45_r Statistics]:")
    print(f"    Min:    {au45.min():.4f}")
    print(f"    Max:    {au45.max():.4f}")
    print(f"    Mean:   {au45.mean():.4f}")
    print(f"    Std:    {au45.std():.4f}")
    print(f"    Q1:     {np.percentile(au45, 25):.4f}")
    print(f"    Median: {np.percentile(au45, 50):.4f}")
    print(f"    Q3:     {np.percentile(au45, 75):.4f}")

    print(f"\n  [Confidence Statistics]:")
    print(f"    Min:    {conf.min():.4f}")
    print(f"    Max:    {conf.max():.4f}")
    print(f"    Mean:   {conf.mean():.4f}")
    print(separator)

    return {
        'label':      label,
        'n_frames':   total_frames,
        'ear_mean':   ears.mean(),
        'ear_std':    ears.std(),
        'ear_min':    ears.min(),
        'ear_max':    ears.max(),
        'ear_q1':     np.percentile(ears, 25),
        'ear_median': np.percentile(ears, 50),
        'ear_q3':     np.percentile(ears, 75),
        'au45_mean':  au45.mean(),
        'au45_std':   au45.std(),
        'au45_q1':    np.percentile(au45, 25),
        'au45_q3':    np.percentile(au45, 75),
        'au45_max':   au45.max(),
    }


# ── Threshold Recommendation ──────────────────────────────────────────────────

def recommend_thresholds(awake_stats, sleeping_stats):
    """Compute and display recommended thresholds."""

    # EAR threshold: midpoint between sleeping mean and awake Q1
    ear_threshold = (sleeping_stats['ear_mean'] + awake_stats['ear_q1']) / 2.0

    # AU45 threshold: awake Q3 + one std to sit above normal blink range
    au45_threshold = awake_stats['au45_q3'] + awake_stats['au45_std']

    # Sleep frame threshold: 2 seconds at 30 FPS
    sleep_frame_threshold = 60

    ear_gap = awake_stats['ear_mean'] - sleeping_stats['ear_mean']

    separator = "=" * 70
    print(f"\n{separator}")
    print(f" RECOMMENDED THRESHOLDS")
    print(separator)

    print(f"\n  EAR Distribution Gap:")
    print(f"    Awake    mean ± std : {awake_stats['ear_mean']:.4f} ± {awake_stats['ear_std']:.4f}")
    print(f"    Sleeping mean ± std : {sleeping_stats['ear_mean']:.4f} ± {sleeping_stats['ear_std']:.4f}")
    print(f"    Gap (awake - sleeping): {ear_gap:.4f}")

    print(f"\n  AU45 Distribution:")
    print(f"    Awake Q3  : {awake_stats['au45_q3']:.4f}")
    print(f"    Awake std : {awake_stats['au45_std']:.4f}")

    print(f"\n  Separation Quality:")
    if ear_gap > 0.05:
        quality = "GOOD — thresholds should work reliably"
        symbol  = "✓"
    elif ear_gap > 0.02:
        quality = "MODERATE — consider increasing SLEEP_FRAME_THRESHOLD to 90"
        symbol  = "⚠"
    else:
        quality = "POOR — EAR alone unreliable, AU45 carries more weight"
        symbol  = "✗"
    print(f"    {symbol} {quality}")

    print(f"\n{separator}")
    print(f" COPY THESE VALUES INTO sleep_detector.py")
    print(separator)
    print(f"\n  EAR_THRESHOLD         = {ear_threshold:.4f}")
    print(f"  AU45_THRESHOLD        = {au45_threshold:.4f}")
    print(f"  SLEEP_FRAME_THRESHOLD = {sleep_frame_threshold}")
    print(f"\n{separator}\n")

    return ear_threshold, au45_threshold, sleep_frame_threshold


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Record awake/sleeping sessions and compute sleep detection thresholds."
    )
    parser.add_argument(
        '--awake-duration', type=int, default=AWAKE_DURATION_SEC,
        help=f'Awake recording duration in seconds (default: {AWAKE_DURATION_SEC})'
    )
    parser.add_argument(
        '--sleep-duration', type=int, default=SLEEPING_DURATION_SEC,
        help=f'Sleeping recording duration in seconds (default: {SLEEPING_DURATION_SEC})'
    )
    parser.add_argument(
        '--awake-csv', type=str, default=None,
        help='Skip awake recording and use existing CSV path'
    )
    parser.add_argument(
        '--sleeping-csv', type=str, default=None,
        help='Skip sleeping recording and use existing CSV path'
    )
    args = parser.parse_args()

    # Validate OpenFace
    if not os.path.exists(OPENFACE_EXE):
        print(f"ERROR: OpenFace not found at {OPENFACE_EXE}")
        print("Please update OPENFACE_DIR at the top of this script.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(" SLEEP DETECTION CALIBRATION TOOL")
    print("=" * 70)
    print("\nThis tool records two sessions:")
    print("  1. AWAKE   — sit normally, look at screen naturally")
    print("  2. SLEEPING — close your eyes, bow your head slightly")
    print(f"\nEach session is {args.awake_duration}s long.")
    print("\nMake sure:")
    print("  • Your face is clearly visible to the webcam")
    print("  • Room lighting is similar to your normal study environment")
    print("  • No other applications are using the webcam")

    # ── Session 1: Awake ──
    if args.awake_csv:
        print(f"\n[AWAKE] Using existing CSV: {args.awake_csv}")
        awake_csv = args.awake_csv
    else:
        print("\n" + "-" * 70)
        print(" SESSION 1: AWAKE")
        print("-" * 70)
        print("Sit naturally, look at your screen as if attending an online class.")
        input("\nPress ENTER when ready to begin awake recording...")

        show_countdown("GET READY — AWAKE SESSION", seconds=3, color=(0, 255, 255))

        awake_csv = record_with_openface(
            output_dir=OUTPUT_BASE_DIR,
            filename_prefix="calibration_awake",
            duration_sec=args.awake_duration,
            label="AWAKE",
            color=(0, 255, 0)
        )

        if awake_csv is None:
            print("ERROR: Awake recording failed. Exiting.")
            sys.exit(1)

    # ── Session 2: Sleeping ──
    if args.sleeping_csv:
        print(f"\n[SLEEPING] Using existing CSV: {args.sleeping_csv}")
        sleeping_csv = args.sleeping_csv
    else:
        print("\n" + "-" * 70)
        print(" SESSION 2: SLEEPING")
        print("-" * 70)
        print("Close your eyes and bow your head slightly,")
        print("simulating falling asleep at your desk.")
        input("\nPress ENTER when ready to begin sleeping recording...")

        show_countdown("GET READY — SLEEPING SESSION", seconds=3, color=(0, 0, 255))

        sleeping_csv = record_with_openface(
            output_dir=OUTPUT_BASE_DIR,
            filename_prefix="calibration_sleeping",
            duration_sec=args.sleep_duration,
            label="SLEEPING",
            color=(0, 0, 255)
        )

        if sleeping_csv is None:
            print("ERROR: Sleeping recording failed. Exiting.")
            sys.exit(1)

    # ── Analysis ──
    print("\n" + "-" * 70)
    print(" ANALYZING RECORDINGS...")
    print("-" * 70)

    awake_stats   = analyze_csv(awake_csv,   label="awake")
    sleeping_stats = analyze_csv(sleeping_csv, label="sleeping")

    if awake_stats is None or sleeping_stats is None:
        print("\nERROR: Analysis failed. Check CSV files and try again.")
        sys.exit(1)

    # ── Thresholds ──
    recommend_thresholds(awake_stats, sleeping_stats)


if __name__ == "__main__":
    main()