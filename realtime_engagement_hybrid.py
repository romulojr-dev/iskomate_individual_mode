"""
Hybrid Pipeline: OpenFace (preferred) with MediaPipe Fallback
==============================================================

If OpenFace is unavailable, automatically switches to MediaPipe for realtime testing.
This allows you to:
    1. Test TCCT-Net inference pipeline immediately (using MediaPipe)
    2. Upgrade to OpenFace later for production (uses correct feature domain)
"""

import cv2
import sys
import os
import json
import torch
import time
import numpy as np
import pandas as pd
import tempfile
import subprocess
import mediapipe as mp
import threading
from queue import Queue
from collections import deque
from pathlib import Path
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'TCCT_Net'))

from TCCT_Net.models.feature_fusion import Decision_Fusion
from TCCT_Net.data.data_processing import batch_cwt
from sleep_detector import SleepDetector, get_final_engagement

# ==================== CONFIG ====================
# TUNING PROFILES for responsiveness
# ⚠️  WINDOW_SIZE MUST STAY AT 280 - Model requirement!
#
PROFILE = "balanced"  # "fast", "balanced", or "accurate"

WINDOW_SIZE = 280  # FIXED - Cannot change (model architecture)

if PROFILE == "fast":
    CHUNK_SIZE = 15
    FRAME_SKIP = 3
    FRAMES_BETWEEN_PREDICTIONS = 5
    RESIZE_SCALE = 0.5
elif PROFILE == "accurate":
    CHUNK_SIZE = 30
    FRAME_SKIP = 1
    FRAMES_BETWEEN_PREDICTIONS = 10
    RESIZE_SCALE = 1.0
else:  # balanced
    CHUNK_SIZE = 20
    FRAME_SKIP = 2
    FRAMES_BETWEEN_PREDICTIONS = 8
    RESIZE_SCALE = 0.7

ENGAGEMENT_LABELS = {
    0: ("Not Engaged",    (0, 0, 255)),
    1: ("Barely Engaged", (0, 165, 255)),
    2: ("Engaged",        (0, 255, 255)),
    3: ("Highly Engaged", (0, 255, 0)),
}
AMPLIFICATION_FACTOR = 2.0
SMOOTHING = 0.85

# OpenFace path (update if you have it installed)
OPENFACE_BIN = r"C:\OpenFace\FeatureExtraction.exe"


# ==================== MEDIAPIPE EXTRACTOR (Fallback) ====================
class MediaPipeExtractor:
    """Fallback extractor using MediaPipe (faster but domain mismatch with training)."""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.9
        )
        self.last_pitch = None
        self.last_yaw = None
    
    def extract_from_frame(self, frame):
        """Extract head pose from single frame."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None, None

        landmarks = results.multi_face_landmarks[0].landmark

        # Stable landmarks
        nose_tip    = np.array([landmarks[4].x * w,   landmarks[4].y * h])
        chin        = np.array([landmarks[152].x * w,  landmarks[152].y * h])
        left_eye    = np.array([landmarks[33].x * w,   landmarks[33].y * h])
        right_eye   = np.array([landmarks[263].x * w,  landmarks[263].y * h])

        # Pitch calculation
        eye_center  = (left_eye + right_eye) / 2
        vertical_vec = chin - eye_center
        pitch = np.arctan2(vertical_vec[0], vertical_vec[1])

        # Yaw calculation
        left_dist  = np.linalg.norm(nose_tip - left_eye)
        right_dist = np.linalg.norm(nose_tip - right_eye)
        if (left_dist + right_dist) > 0:
            yaw = np.arctan2(right_dist - left_dist, (right_dist + left_dist) / 2)
        else:
            yaw = 0.0

        # Clamp and smooth
        pitch = np.clip(-pitch, -1.5, 1.5)
        yaw   = np.clip(yaw,    -1.5, 1.5)

        # Exponential smoothing
        if self.last_pitch is None:
            self.last_pitch = pitch
            self.last_yaw   = yaw
        else:
            self.last_pitch = SMOOTHING * self.last_pitch + (1 - SMOOTHING) * pitch
            self.last_yaw   = SMOOTHING * self.last_yaw   + (1 - SMOOTHING) * yaw

        return self.last_pitch, self.last_yaw
    
    def cleanup(self):
        self.face_mesh.close()


# ==================== OPENFACE EXTRACTOR ====================
class OpenFaceExtractor:
    """Batch OpenFace extractor (slower but correct feature domain)."""
    
    def __init__(self, openface_bin=OPENFACE_BIN, fps=30):
        self.openface_bin = openface_bin
        self.fps = fps
        self.temp_dir = tempfile.mkdtemp(prefix="openface_")
        self.available = os.path.exists(self.openface_bin)
    
    def extract_from_frames(self, frame_list):
        """Extract pose and full data from frame list using OpenFace."""
        if not self.available or len(frame_list) == 0:
            return None, None
        
        try:
            video_path = os.path.join(self.temp_dir, f"chunk_{int(time.time() * 1000)}.mp4")
            self._write_video(frame_list, video_path)
            csv_path = self._run_openface(video_path)
            
            if csv_path and os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if ' pose_Rx' in df.columns and ' pose_Ry' in df.columns:
                    pose_result = df[[' pose_Rx', ' pose_Ry']].copy()
                    full_result = df  # Return full dataframe for sleep detection
                    os.remove(video_path)
                    os.remove(csv_path)
                    return pose_result, full_result
        except Exception as e:
            print(f"[OpenFace Error] {e}")
        
        return None, None
    
    def _write_video(self, frame_list, output_path):
        if len(frame_list) == 0:
            return
        h, w = frame_list[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))
        for frame in frame_list:
            writer.write(frame)
        writer.release()
    
    def _run_openface(self, video_path):
        output_dir = r"C:\OpenFace\output_au"
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = [
            self.openface_bin,
            "-f", video_path,
            "-out_dir", output_dir,
            "-device", "0",
            "-pose",
            "-3Dfp",  # 3D facial points for sleep detection
            "-aus",   # Action units for sleep detection
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=30, text=True)
            if result.returncode != 0:
                return None
            csv_files = list(Path(output_dir).glob("*.csv"))
            return str(csv_files[0]) if csv_files else None
        except:
            return None
    
    def cleanup(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


# ==================== BACKGROUND WORKER ====================
class OpenFaceBackgroundWorker:
    """
    Runs OpenFace extraction in a background thread to prevent video feed from freezing.
    Producer-Consumer Pattern:
    - Main thread: Captures frames, displays video
    - Worker thread: Processes frames through OpenFace asynchronously
    """
    
    def __init__(self, openface_extractor):
        self.extractor = openface_extractor
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.running = False
        self.worker_thread = None
        self.processing = False
    
    def start(self):
        """Start background worker thread."""
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print("[Worker] Background thread started")
    
    def stop(self):
        """Stop background worker thread."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2)
        print("[Worker] Background thread stopped")
    
    def submit_task(self, frame_buffer):
        """Submit a frame buffer for OpenFace processing (non-blocking)."""
        self.task_queue.put(frame_buffer)
        self.processing = True
    
    def get_result(self):
        """Get completed result from OpenFace (non-blocking)."""
        if self.result_queue.empty():
            return None, None, None
        return self.result_queue.get_nowait()
    
    def _worker_loop(self):
        """Background worker loop - runs in separate thread."""
        while self.running:
            try:
                frame_buffer = self.task_queue.get(timeout=0.5)
                if frame_buffer is None:
                    break
                
                pose_df, full_df = self.extractor.extract_from_frames(frame_buffer)
                self.result_queue.put((frame_buffer, pose_df, full_df))
                
                # IMPORTANT: Reset flag so next batch can be submitted
                self.processing = False
                
            except:
                # Queue timeout or error, reset processing flag
                self.processing = False


# ==================== TCCT-NET INFERENCE ====================
def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def load_model(config):
    device = torch.device("cpu")
    model = Decision_Fusion(config['n_classes'])
    model = nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(torch.load(
        os.path.join('TCCT_Net', config['final_model_weights']),
        map_location=device
    ))
    model.eval()
    return model, device


def predict_engagement(model, device, pose_buffer, config):
    """Predict engagement from pose buffer."""
    data = list(pose_buffer)
    while len(data) < WINDOW_SIZE:
        data.insert(0, data[0] if data else [0.0, 0.0])
    
    df = pd.DataFrame(data, columns=[" pose_Rx", " pose_Ry"])
    
    # ==================== CRITICAL: Z-SCORE NORMALIZATION ====================
    # Normalize using training data statistics (from config)
    df = (df - config['target_mean']) / config['target_std']
    
    df = df * AMPLIFICATION_FACTOR
    weights = np.linspace(0.5, 1.5, WINDOW_SIZE)
    df[" pose_Rx"] = df[" pose_Rx"] * weights
    df[" pose_Ry"] = df[" pose_Ry"] * weights
    
    signal = torch.tensor(df.values, dtype=torch.float32)
    signal = signal.T.unsqueeze(0).unsqueeze(0)
    signal = signal.to(device)
    
    frequencies = np.linspace(
        config['freq_min'],
        config['freq_max'],
        config['tensor_height']
    )
    
    with torch.no_grad():
        cwt = batch_cwt(signal, frequencies, sampling_frequency=config['sampling_frequency'])
        output = model(signal, cwt)
        predicted_class = output.argmax(dim=1).item()
    
    return predicted_class


# ==================== MAIN HYBRID PIPELINE ====================
def main():
    print("\n" + "=" * 70)
    print("Hybrid Pipeline: OpenFace (preferred) + MediaPipe (fallback)")
    print("=" * 70 + "\n")
    
    # Load config and model
    config = load_config('TCCT_Net/config.json')
    model, device = load_model(config)
    print("[TCCT-Net] Model loaded")
    
    # Try OpenFace first
    openface = OpenFaceExtractor(OPENFACE_BIN)
    use_openface = openface.available
    
    if use_openface:
        print(f"[Extractor] Using OpenFace: {OPENFACE_BIN}")
        print("            (Features match TCCT-Net training domain)")
        # Create background worker for OpenFace
        worker = OpenFaceBackgroundWorker(openface)
        worker.start()
    else:
        print("[Extractor] OpenFace not found, falling back to MediaPipe")
        print("            (Warning: domain mismatch with TCCT-Net training)")
        mediapose = MediaPipeExtractor()
        worker = None
    
    # Pose buffer
    pose_buffer = deque(maxlen=WINDOW_SIZE)
    frame_buffer = []
    
    # Sleep detector (EAR-only approach with empirically tuned thresholds)
    sleep_detector = SleepDetector()  # Uses defaults: EAR_THRESHOLD=0.4920, SLEEP_FRAME_THRESHOLD=90
    
    engagement_class = None
    frame_count = 0
    pred_count = 0
    last_pred_frame = -999  # Track frame number of last prediction
    buffer_initialized = False  # Track if buffer is full (steady state)
    processing_status = "Ready"
    
    # Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("[Camera] Opened\n")
    
    prev_time = time.time()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Initialize sleep detection state (will be updated if OpenFace results available)
            is_sleeping = False
            
            # ==================== CHECK FOR OPENFACE RESULTS ====================
            # This is non-blocking and happens every frame
            if use_openface and worker is not None:
                pose_buffer_result, pose_df, full_df = worker.get_result()
                
                if pose_df is not None and len(pose_df) > 0:
                    print(f"[OpenFace] OK Extracted {len(pose_df)} pose samples")
                    
                    # Add new poses to buffer
                    for idx, row in pose_df.iterrows():
                        pose_buffer.append([row[' pose_Rx'], row[' pose_Ry']])
                    
                    processing_status = "Ready"
                    
                    # ==================== SLEEP DETECTION (First Priority) ====================
                    # Process each frame through sleep detector (EAR-only approach)
                    is_sleeping = False
                    ear_val = None
                    
                    if full_df is not None and len(full_df) > 0:
                        # Process each frame in the batch through the sleep detector
                        for frame_idx, (idx, row) in enumerate(full_df.iterrows()):
                            is_sleeping, ear_val, _ = sleep_detector.update(row)
                        
                        perclos = sleep_detector.get_perclos()
                        consec_frames = sleep_detector.consecutive_closed_frames
                        print(f"[Sleep Detection] EAR={ear_val:.4f} (threshold={sleep_detector.ear_threshold:.4f}), Consecutive={consec_frames}/90, PERCLOS={perclos:.1%}, Sleep={is_sleeping}")
                    
                    # ==================== RUN PREDICTIONS (TIED TO DATA ARRIVAL) ====================
                    # Only predict when buffer is full and we have fresh OpenFace data
                    if len(pose_buffer) >= WINDOW_SIZE:
                        frames_since_last_pred = frame_count - last_pred_frame
                        
                        if pred_count == 0 or frames_since_last_pred >= FRAMES_BETWEEN_PREDICTIONS:
                            try:
                                # LOGIC FLOW: Sleep check takes absolute priority
                                if is_sleeping:
                                    engagement_class, status = get_final_engagement(None, is_sleeping=True)
                                    print(f"[Sleep Detection] EAR-based sleep detected. Engagement: {status}")
                                else:
                                    tcct_pred = predict_engagement(model, device, pose_buffer, config)
                                    engagement_class, status = get_final_engagement(tcct_pred, is_sleeping=False)
                                
                                pred_count += 1
                                last_pred_frame = frame_count
                                label_text, _ = ENGAGEMENT_LABELS[engagement_class]
                                print(f"[Prediction #{pred_count}] Engagement: {label_text}")
                            except Exception as e:
                                print(f"[Inference Error] {e}")
            
            # Skip frames for faster processing
            if frame_count % FRAME_SKIP != 0:
                # Still display skipped frames for smooth video
                display_frame = cv2.resize(frame, (640, 480))
                h, w = display_frame.shape[:2]
                
                extractor_name = "OpenFace (accurate)" if use_openface else "MediaPipe (fast)"
                cv2.putText(display_frame, f"Extractor: {extractor_name}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
                
                if use_openface:
                    status_color = (0, 255, 0) if processing_status == "Ready" else (0, 165, 255)
                    cv2.putText(display_frame, f"Status: {processing_status}",
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
                
                buffer_pct = int((len(pose_buffer) / WINDOW_SIZE) * 100)
                cv2.putText(display_frame, f"Buffer: {buffer_pct}%",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                
                if engagement_class is not None:
                    label_text, color = ENGAGEMENT_LABELS[engagement_class]
                    cv2.putText(display_frame, f"Engagement: {label_text}",
                               (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.rectangle(display_frame, (5, 130), (w - 5, 170), color, 3)
                
                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                cv2.putText(display_frame, f"FPS: {fps:.1f}",
                           (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow("Hybrid Engagement Detection", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Resize for processing
            frame = cv2.resize(frame, (int(640 * RESIZE_SCALE), int(480 * RESIZE_SCALE)))
            frame_buffer.append(frame.copy())
            
            # ==================== DETERMINE CHUNK SIZE MODE ====================
            # WARMUP PHASE: Buffer is not full yet, collect batches
            # STEADY STATE: Buffer is full, use sliding window (collect CHUNK_SIZE frames for temporal context)
            if use_openface:
                if len(pose_buffer) < WINDOW_SIZE:
                    # Warmup mode: collect CHUNK_SIZE frames before submitting
                    chunk_threshold = CHUNK_SIZE
                else:
                    # Steady state: buffer full, collect CHUNK_SIZE frames for next batch (sliding window)
                    buffer_initialized = True
                    chunk_threshold = CHUNK_SIZE
                
                # ==================== SUBMIT OPENFACE TASK (NON-BLOCKING) ====================
                if len(frame_buffer) >= chunk_threshold and not worker.processing:
                    if buffer_initialized:
                        print(f"\n[Frame {frame_count}] Sliding window: Submitting {len(frame_buffer)} frames...")
                    else:
                        print(f"\n[Frame {frame_count}] Warmup: Submitting {len(frame_buffer)} frames to OpenFace...")
                    
                    worker.submit_task(frame_buffer)
                    frame_buffer = []
                    processing_status = "Processing OpenFace..."
            
            # MediaPipe: per-frame extraction (non-blocking, runs on main thread)
            elif not use_openface and len(frame_buffer) > 0:
                pitch, yaw = mediapose.extract_from_frame(frame_buffer[-1])
                if pitch is not None:
                    pose_buffer.append([pitch, yaw])
                    
                    # ==================== RUN PREDICTIONS (TIED TO DATA ARRIVAL) ====================
                    # Predict when buffer is full and new data just arrived
                    if len(pose_buffer) >= WINDOW_SIZE:
                        frames_since_last_pred = frame_count - last_pred_frame
                        
                        if pred_count == 0 or frames_since_last_pred >= FRAMES_BETWEEN_PREDICTIONS:
                            try:
                                engagement_class = predict_engagement(model, device, pose_buffer, config)
                                pred_count += 1
                                last_pred_frame = frame_count
                                label_text, _ = ENGAGEMENT_LABELS[engagement_class]
                                print(f"[Prediction #{pred_count}] {label_text}")
                            except Exception as e:
                                print(f"[Inference Error] {e}")
            
            # Render (resize back to standard display size)
            display_frame = cv2.resize(frame, (640, 480))
            h, w = display_frame.shape[:2]
            
            # Status bar
            extractor_name = "OpenFace (accurate)" if use_openface else "MediaPipe (fast)"
            cv2.putText(display_frame, f"Extractor: {extractor_name}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
            
            if use_openface:
                status_color = (0, 255, 0) if processing_status == "Ready" else (0, 165, 255)
                cv2.putText(display_frame, f"Status: {processing_status}",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                
                # Chunk status - show current threshold
                current_threshold = 1 if buffer_initialized else CHUNK_SIZE
                phase_label = "Sliding Window" if buffer_initialized else "Warmup"
                cv2.putText(display_frame, f"Frame Chunk: {len(frame_buffer)}/{current_threshold} ({phase_label})",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            buffer_pct = int((len(pose_buffer) / WINDOW_SIZE) * 100)
            cv2.putText(display_frame, f"Buffer: {buffer_pct}%",
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # Engagement
            if engagement_class is not None:
                label_text, color = ENGAGEMENT_LABELS[engagement_class]
                cv2.putText(display_frame, f"Engagement: {label_text}",
                           (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.rectangle(display_frame, (5, 160), (w - 5, 200), color, 3)
            else:
                cv2.putText(display_frame, "Warming up...",
                           (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
            
            # FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            cv2.putText(display_frame, f"FPS: {fps:.1f}",
                       (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Hybrid Engagement Detection", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        if use_openface:
            if worker is not None:
                worker.stop()
            openface.cleanup()
        else:
            mediapose.cleanup()
        
        print("\n[Pipeline] Closed")


if __name__ == "__main__":
    main()
