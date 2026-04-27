"""
Automated EAR + AU45 Threshold Calibration Script

Usage:
    python calibrate_ear_au45.py [awake|sleeping]

Examples:
    python calibrate_ear_au45.py awake
    python calibrate_ear_au45.py sleeping

This script:
1. Opens a camera and records video (press SPACE to start, records ~10 seconds)
2. Automatically processes video through OpenFace
3. Computes EAR from eye landmarks
4. Extracts AU45_r values
5. Prints comprehensive statistics to help calibrate thresholds
6. Saves results to calibration_<state>_results.txt
"""

import cv2
import pandas as pd
import numpy as np
import sys
import time
import tempfile
import subprocess
import os
from pathlib import Path

OPENFACE_BIN = r"C:\OpenFace\FeatureExtraction.exe"

def write_video(frame_list, output_path, fps=30):
    """Write frames to video file."""
    if len(frame_list) == 0:
        return
    h, w = frame_list[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frame_list:
        writer.write(frame)
    writer.release()


def run_openface(video_path, temp_dir):
    """Run OpenFace on video and return CSV path."""
    # Create standard OpenFace output directory
    output_dir = r"C:\OpenFace\output_au"
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        OPENFACE_BIN,
        "-f", video_path,
        "-out_dir", output_dir,
        "-device", "0",
        "-pose",
        "-3Dfp",
        "-aus",
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60, text=True)
        if result.returncode != 0:
            print(f"[OpenFace Error] {result.stderr}")
            return None
        csv_files = list(Path(output_dir).glob("*.csv"))
        return str(csv_files[0]) if csv_files else None
    except Exception as e:
        print(f"[OpenFace Exception] {e}")
        return None


def compute_ear_from_csv(csv_path):
    """
    Compute EAR for all valid frames in an OpenFace CSV.
    
    Returns:
        (ear_values, au45_values, confidences)
    """
    df = pd.read_csv(csv_path)
    
    ear_values = []
    au45_values = []
    confidences = []
    
    for idx, row in df.iterrows():
        # Check validity
        try:
            success = int(row[' success'])
            confidence = float(row.get(' confidence', 1.0))
            
            if success != 1 or confidence < 0.5:
                continue
        except:
            continue
        
        # Extract AU45_r
        try:
            au45 = float(row.get(' AU45_r', 0.0))
        except:
            au45 = 0.0
        
        # Extract eye landmarks (eye_lmk_x_i / eye_lmk_y_i format)
        try:
            # Left eye key points
            lx = [float(row[f' eye_lmk_x_{i}']) for i in [0, 1, 7, 8, 9, 13]]
            ly = [float(row[f' eye_lmk_y_{i}']) for i in [0, 1, 7, 8, 9, 13]]
            
            # Right eye key points
            rx = [float(row[f' eye_lmk_x_{i}']) for i in [28, 29, 35, 36, 37, 41]]
            ry = [float(row[f' eye_lmk_y_{i}']) for i in [28, 29, 35, 36, 37, 41]]
            
            # Compute EAR
            def ear(x, y):
                v1 = np.sqrt((x[1]-x[5])**2 + (y[1]-y[5])**2)
                v2 = np.sqrt((x[2]-x[4])**2 + (y[2]-y[4])**2)
                h  = np.sqrt((x[0]-x[3])**2 + (y[0]-y[3])**2)
                return (v1 + v2) / (2.0 * h) if h > 0 else 0.3
            
            left_ear = ear(lx, ly)
            right_ear = ear(rx, ry)
            avg_ear = (left_ear + right_ear) / 2.0
            
            ear_values.append(avg_ear)
            au45_values.append(au45)
            confidences.append(confidence)
            
        except Exception as e:
            # Skip frames with missing eye landmarks
            continue
    
    return np.array(ear_values), np.array(au45_values), np.array(confidences)


def print_stats(label, values):
    """Print comprehensive statistics for a value array."""
    if len(values) == 0:
        print(f"  {label}: No valid data")
        return []
    
    stats_list = [
        f"  {label} Statistics ({len(values)} frames):",
        f"    Min:     {np.min(values):.4f}",
        f"    Max:     {np.max(values):.4f}",
        f"    Mean:    {np.mean(values):.4f}",
        f"    Std:     {np.std(values):.4f}",
        f"    Median:  {np.median(values):.4f}",
        f"    Q1:      {np.percentile(values, 25):.4f}",
        f"    Q3:      {np.percentile(values, 75):.4f}",
    ]
    
    for line in stats_list:
        print(line)
    
    return stats_list


def main():
    if len(sys.argv) < 2:
        print("Usage: python calibrate_ear_au45.py [awake|sleeping]")
        print("\nExample:")
        print("  python calibrate_ear_au45.py awake")
        print("  python calibrate_ear_au45.py sleeping")
        return
    
    state = sys.argv[1].lower()
    if state not in ['awake', 'sleeping']:
        print(f"[ERROR] Invalid state: {state}. Use 'awake' or 'sleeping'")
        return
    
    output_file = f"calibration_{state}_results.txt"
    
    print("\n" + "=" * 70)
    print(f"AUTOMATED EAR + AU45 CALIBRATION - {state.upper()}")
    print("=" * 70)
    print("\nInstructions:")
    if state == 'awake':
        print("  - Stay WIDE AWAKE, eyes OPEN, natural gaze the entire time")
        print("  - Blink normally, do NOT force eyes closed")
    else:
        print("  - CLOSE YOUR EYES or keep them mostly closed")
        print("  - Simulate drowsy/sleeping state")
    print("  - Press SPACE to START recording")
    print("  - Automatic stop after ~10 seconds")
    print("=" * 70 + "\n")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return
    
    cap.set(cv2.CAP_PROP_FPS, 30)
    recording = False
    frames = []
    start_time = None
    
    print("Press SPACE to start, 'q' to quit\n")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        display_frame = cv2.resize(frame, (640, 480))
        
        if recording:
            elapsed = time.time() - start_time
            frames.append(frame)
            
            status = f"RECORDING: {elapsed:.1f}s / 10s"
            cv2.putText(display_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"State: {state.upper()}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            
            if elapsed >= 10:
                recording = False
                print(f"\n[Calibration] Captured {len(frames)} frames")
                break
        else:
            cv2.putText(display_frame, "Press SPACE to start", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.putText(display_frame, f"State: {state.upper()}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        cv2.imshow("EAR+AU45 Calibration", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            recording = True
            frames = []
            start_time = time.time()
            print(f"[Calibration] Recording {state.upper()} state...")
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(frames) == 0:
        print("[ERROR] No frames captured")
        return
    
    print("\n[OpenFace] Processing frames...")
    temp_dir = tempfile.mkdtemp(prefix="calibrate_")
    video_path = os.path.join(temp_dir, "calibrate.mp4")
    write_video(frames, video_path)
    
    csv_path = run_openface(video_path, temp_dir)
    if not csv_path:
        print("[ERROR] OpenFace processing failed")
        return
    
    print("[Analysis] Analyzing EAR and AU45 values...")
    
    try:
        ear_vals, au45_vals, conf_vals = compute_ear_from_csv(csv_path)
        
        if len(ear_vals) == 0:
            print("[ERROR] No valid frames found in CSV")
            return
        
        # Collect all output lines
        output_lines = []
        output_lines.append("=" * 70)
        output_lines.append(f"EAR + AU45 CALIBRATION - {state.upper()}")
        output_lines.append("=" * 70)
        output_lines.append(f"Frames analyzed: {len(ear_vals)}")
        output_lines.append("")
        
        # Print and collect stats
        print("\n" + "=" * 70)
        print(f"CALIBRATION RESULTS - {state.upper()}")
        print("=" * 70 + "\n")
        
        stats_ear = print_stats("EAR (Eye Aspect Ratio)", ear_vals)
        output_lines.extend(stats_ear)
        output_lines.append("")
        
        stats_au45 = print_stats("AU45_r (Eye Closure Action Unit)", au45_vals)
        output_lines.extend(stats_au45)
        output_lines.append("")
        
        stats_conf = print_stats("Confidence", conf_vals)
        output_lines.extend(stats_conf)
        
        # Additional analysis
        print("\n" + "=" * 70)
        print("THRESHOLD ANALYSIS")
        print("=" * 70)
        
        output_lines.append("\n" + "=" * 70)
        output_lines.append("THRESHOLD ANALYSIS")
        output_lines.append("=" * 70)
        
        # Recommend EAR threshold
        ear_mean = np.mean(ear_vals)
        ear_std = np.std(ear_vals)
        ear_threshold_recommended = ear_mean - (1.5 * ear_std)
        
        print(f"\nEAR Threshold:")
        print(f"  Current setting:      0.20")
        print(f"  Recommended:          {ear_threshold_recommended:.4f} (mean - 1.5*std)")
        print(f"  Q1 (25th percentile): {np.percentile(ear_vals, 25):.4f}")
        print(f"  Q3 (75th percentile): {np.percentile(ear_vals, 75):.4f}")
        
        output_lines.append(f"\nEAR Threshold:")
        output_lines.append(f"  Current setting:      0.20")
        output_lines.append(f"  Recommended:          {ear_threshold_recommended:.4f} (mean - 1.5*std)")
        output_lines.append(f"  Q1 (25th percentile): {np.percentile(ear_vals, 25):.4f}")
        output_lines.append(f"  Q3 (75th percentile): {np.percentile(ear_vals, 75):.4f}")
        
        # Recommend AU45 threshold
        au45_above_05 = (au45_vals > 0.5).sum()
        au45_above_10 = (au45_vals > 1.0).sum()
        au45_above_20 = (au45_vals > 2.0).sum()
        
        print(f"\nAU45_r Threshold:")
        print(f"  Current setting:   0.50")
        print(f"  Frames > 0.5:      {au45_above_05} ({100*au45_above_05/len(au45_vals):.1f}%)")
        print(f"  Frames > 1.0:      {au45_above_10} ({100*au45_above_10/len(au45_vals):.1f}%)")
        print(f"  Frames > 2.0:      {au45_above_20} ({100*au45_above_20/len(au45_vals):.1f}%)")
        
        output_lines.append(f"\nAU45_r Threshold:")
        output_lines.append(f"  Current setting:   0.50")
        output_lines.append(f"  Frames > 0.5:      {au45_above_05} ({100*au45_above_05/len(au45_vals):.1f}%)")
        output_lines.append(f"  Frames > 1.0:      {au45_above_10} ({100*au45_above_10/len(au45_vals):.1f}%)")
        output_lines.append(f"  Frames > 2.0:      {au45_above_20} ({100*au45_above_20/len(au45_vals):.1f}%)")
        
        # Both conditions
        both_closed = ((ear_vals < 0.20) & (au45_vals > 0.5)).sum()
        print(f"\nBoth Conditions (EAR < 0.20 AND AU45 > 0.5):")
        print(f"  Frames matching: {both_closed} ({100*both_closed/len(ear_vals):.1f}%)")
        
        output_lines.append(f"\nBoth Conditions (EAR < 0.20 AND AU45 > 0.5):")
        output_lines.append(f"  Frames matching: {both_closed} ({100*both_closed/len(ear_vals):.1f}%)")
        
        # Interpretation
        print("\n" + "=" * 70)
        print("INTERPRETATION")
        print("=" * 70)
        
        output_lines.append("\n" + "=" * 70)
        output_lines.append("INTERPRETATION")
        output_lines.append("=" * 70)
        
        if state == "awake":
            interpretation = f"""
For AWAKE state:
  - EAR mean: {ear_mean:.4f} (should be HIGH, e.g., > 0.25)
  - AU45 > 0.5: {100*au45_above_05/len(au45_vals):.1f}% (should be LOW, < 5%)
  - Both closed: {100*both_closed/len(ear_vals):.1f}% (should be VERY LOW, < 1%)
  
Interpretation:
  - If EAR is LOW during awake state: Camera angle or landmark issue
  - If AU45 high: False positives, increase AU45 threshold
  - If both closed low: Good baseline, thresholds appropriate
"""
        else:
            interpretation = f"""
For SLEEPING state:
  - EAR mean: {ear_mean:.4f} (should be LOW, e.g., < 0.15)
  - AU45 > 0.5: {100*au45_above_05/len(au45_vals):.1f}% (should be HIGH, > 50%)
  - Both closed: {100*both_closed/len(ear_vals):.1f}% (should be HIGH, > 50%)
  
Interpretation:
  - If EAR is HIGH during sleep: Eyes still somewhat open or movement
  - If AU45 low: AU45 not reliably detecting closure, may need to adjust
  - If both closed high: Good sensitivity, thresholds appropriate
"""
        
        print(interpretation)
        output_lines.append(interpretation)
        
        print("\n" + "=" * 70)
        print(f"NEXT STEP: Run on the other state (awake/sleeping) and compare")
        print("=" * 70 + "\n")
        
        output_lines.append("\n" + "=" * 70)
        output_lines.append(f"NEXT STEP: Run on the other state and compare")
        output_lines.append("=" * 70)
        
        # Save to file
        output_text = "\n".join(output_lines)
        with open(output_file, 'w') as f:
            f.write(output_text)
        
        print(f"[Saved] Results to {output_file}")
        
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


if __name__ == "__main__":
    main()

