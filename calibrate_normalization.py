"""
Run this script while sitting normally in front of your webcam for 60 seconds.
It will compute the correct mean and std for YOUR MediaPipe setup and update config.json.
"""
import cv2
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from mediapipe_extractor import HeadPoseExtractor

def calibrate(duration_seconds=60):
    extractor = HeadPoseExtractor()
    cap = cv2.VideoCapture(0)

    all_values = []
    fps = 30
    total_frames = duration_seconds * fps
    frame_count = 0

    print(f"Calibrating for {duration_seconds} seconds.")
    print("Sit naturally in front of your webcam as you normally would when studying.")
    print("Press 'q' to stop early.")

    while cap.isOpened() and frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        pitch, yaw, roll, annotated = extractor.get_head_pose(frame)

        if pitch is not None:
            all_values.append([pitch, yaw])
            frame_count += 1
            progress = int((frame_count / total_frames) * 100)
            cv2.putText(annotated, f"Calibrating: {progress}%", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Calibration - Sit naturally", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.release()

    if len(all_values) < 10:
        print("Not enough data collected. Try again.")
        return

    arr = np.array(all_values)
    mean = float(np.mean(arr))
    std  = float(np.std(arr))

    print(f"\nCalibration complete.")
    print(f"Frames collected : {len(all_values)}")
    print(f"Computed mean    : {mean:.6f}")
    print(f"Computed std     : {std:.6f}")

    # Update config.json
    config_path = 'TCCT_Net/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    old_mean = config['target_mean']
    old_std  = config['target_std']
    config['target_mean'] = mean
    config['target_std']  = std

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"\nconfig.json updated:")
    print(f"  target_mean: {old_mean} → {mean:.6f}")
    print(f"  target_std:  {old_std} → {std:.6f}")
    print("\nYou can now run realtime_engagement.py with improved accuracy.")

if __name__ == "__main__":
    calibrate(duration_seconds=60)