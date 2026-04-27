"""
Realtime Engagement Detection: OpenFace + TCCT-Net Pipeline

Pipeline:
1. Capture webcam frames into a temporary video buffer
2. Every N frames, write buffer to temporary .mp4 file
3. Run OpenFace FeatureExtraction on the temp video
4. Read generated CSV with pose features
5. Feed pose sequence into TCCT-Net model
6. Display engagement prediction in realtime
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

# TUNING PROFILES: Choose one based on your preference
#   "fast"      = responsive live view, lower latency (~3-4s)
#   "balanced"  = good trade-off (~5-6s latency)
#   "accurate"  = more temporal data (~8-10s latency)
#   "custom"    = ~2s latency (lowest possible, may reduce stability)
#
# ⚠️  WINDOW_SIZE MUST STAY AT 280 - This is the model's input size!
#     The TCCT-Net model was trained with exactly 280 frames.
#     Changing this will cause a shape mismatch error.

PROFILE = "custom"  # Change to "fast", "balanced", "accurate", or "custom"

# Fixed: WINDOW_SIZE = 280 (DO NOT CHANGE - hardcoded in model weights)
WINDOW_SIZE = 280

if PROFILE == "fast":
    CHUNK_SIZE = 15         # Process OpenFace more frequently
    FRAME_SKIP = 3          # Skip 2 out of 3 frames (less processing)
    FRAMES_BETWEEN_PREDICTIONS = 5
    RESIZE_SCALE = 0.5      # Resize frames to 50% for faster OpenFace

elif PROFILE == "accurate":
    CHUNK_SIZE = 30         # Larger batches
    FRAME_SKIP = 1          # Process every frame
    FRAMES_BETWEEN_PREDICTIONS = 10
    RESIZE_SCALE = 1.0      # Full resolution

elif PROFILE == "custom":
    CHUNK_SIZE = 10         # Very frequent predictions
    FRAME_SKIP = 2          # Skip every 2nd frame (15 FPS)
    FRAMES_BETWEEN_PREDICTIONS = 3
    RESIZE_SCALE = 0.7      # 70% resolution for balance

else:  # balanced (default)
    CHUNK_SIZE = 20         # Medium batch
    FRAME_SKIP = 2          # Skip every 2nd frame
    FRAMES_BETWEEN_PREDICTIONS = 8
    RESIZE_SCALE = 0.7      # 70% resolution for faster processing

print(f"[Config] Profile: {PROFILE}")
print(f"         Window: {WINDOW_SIZE} (FIXED), Chunk: {CHUNK_SIZE}, Skip: {FRAME_SKIP}, Scale: {RESIZE_SCALE}")

ENGAGEMENT_LABELS = {
    0: ("Not Engaged",    (0, 0, 255)),
    1: ("Barely Engaged", (0, 165, 255)),
    2: ("Engaged",        (0, 255, 255)),
    3: ("Highly Engaged", (0, 255, 0)),
}
AMPLIFICATION_FACTOR = 2.0

# Path to OpenFace FeatureExtraction executable
# Adjust this based on your OpenFace installation
OPENFACE_BIN = r"C:\OpenFace\FeatureExtraction.exe"  # Windows path example
# For Linux/Mac: OPENFACE_BIN = "/path/to/OpenFace/build/bin/FeatureExtraction"


# ==================== OPENFACE EXTRACTOR ====================
class OpenFaceExtractor:
    """
    Extracts features from live camera stream using OpenFace.
    Runs OpenFace on device directly (bypasses codec issues on Windows).
    """
    
    def __init__(self, openface_bin=OPENFACE_BIN, fps=30):
        self.openface_bin = openface_bin
        self.fps = fps
        self.output_dir = r"C:\OpenFace\output_au"
        os.makedirs(self.output_dir, exist_ok=True)
        
        if not os.path.exists(self.openface_bin):
            print(f"[WARNING] OpenFace binary not found at: {self.openface_bin}")
            self.available = False
        else:
            self.available = True
            self.process = None
            self.csv_rows = []
            self._start_openface()
    
    def _start_openface(self):
        """
        Start OpenFace FeatureExtraction process running on live device stream.
        OpenFace owns the device exclusively to avoid codec issues.
        """
        # Use filename that OpenFace will write to
        self.csv_path = os.path.join(self.output_dir, "face_out.csv")
        
        # Remove old CSV if exists
        if os.path.exists(self.csv_path):
            try:
                os.remove(self.csv_path)
            except:
                pass
        
        cmd = [
            self.openface_bin,
            "-device", "0",           # Let OpenFace own the device
            "-out_dir", self.output_dir,
            "-of", "face_out",        # Output filename prefix
            "-pose",                  # Head pose
            "-2Dfp",                  # 2D facial landmarks
            "-3Dfp",                  # 3D facial landmarks
            "-aus",                   # Action units (needed for eye landmarks)
        ]
        
        try:
            print("[OpenFace] Starting live stream processing (device owned by OpenFace)...")
            print(f"[OpenFace] Command: {' '.join(cmd)}")
            
            # Suppress OpenFace console output to keep terminal clean
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1
            )
            print("[OpenFace] Live stream subprocess started - device allocated exclusively")
        except Exception as e:
            print(f"[ERROR] Failed to start OpenFace: {e}")
            self.available = False
    
    def extract_from_frames(self, frame_list):
        """
        Extract features from frame list.
        Since OpenFace runs on live stream, this just reads from CSV.
        
        Args:
            frame_list: list of frames (used only for sync timing, not processing)
            
        Returns:
            (pose_df, full_df) tuple - most recent frame from CSV
        """
        if not self.available or self.process is None:
            return None, None
        
        try:
            # Read latest rows from OpenFace CSV file
            pose_df = None
            full_df = None
            
            # Keep checking for CSV file
            if os.path.exists(self.csv_path):
                try:
                    df = pd.read_csv(self.csv_path)
                    if len(df) > 0:
                        full_df = df
                        
                        # Extract latest pose frame
                        if ' pose_Rx' in df.columns and ' pose_Ry' in df.columns:
                            pose_df = df[[' pose_Rx', ' pose_Ry']].copy()
                        
                        return pose_df, full_df
                except Exception as e:
                    # CSV might be partially written
                    pass
            
            return pose_df, full_df
        
        except Exception as e:
            print(f"[OpenFace Error] {e}")
            import traceback
            traceback.print_exc()
        
        return None, None
    
    def cleanup(self):
        """Terminate OpenFace subprocess and clean up."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                print("[OpenFace] Process terminated")
            except:
                try:
                    self.process.kill()
                    print("[OpenFace] Process killed")
                except:
                    pass


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
        self.task_queue = Queue()  # Queue of frame buffers to process
        self.result_queue = Queue()  # Queue of (frame_buffer, pose_df) results
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
        """
        Submit a frame buffer for OpenFace processing.
        Non-blocking - returns immediately.
        """
        self.task_queue.put(frame_buffer)
        self.processing = True
    
    def get_result(self):
        """
        Get the next completed result from OpenFace.
        Returns (frame_buffer, pose_df, full_df) or (None, None, None) if no result ready.
        """
        if self.result_queue.empty():
            return None, None, None
        return self.result_queue.get_nowait()
    
    def _worker_loop(self):
        """Background worker loop - runs in separate thread."""
        while self.running:
            try:
                # Wait for task (with timeout to allow graceful shutdown)
                frame_buffer = self.task_queue.get(timeout=0.5)
                
                if frame_buffer is None:  # Sentinel value to stop
                    break
                
                # Process frames through OpenFace (blocking, but in background thread)
                pose_df, full_df = self.extractor.extract_from_frames(frame_buffer)
                
                # Put result in queue for main thread to consume
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
    """
    Predict engagement from pose buffer.
    
    Args:
        model: TCCT-Net model
        device: torch device
        pose_buffer: deque of [pose_Rx, pose_Ry] pairs
        config: config dict
        
    Returns:
        predicted engagement class (0-3)
    """
    data = list(pose_buffer)
    while len(data) < WINDOW_SIZE:
        data.insert(0, data[0] if data else [0.0, 0.0])
    
    df = pd.DataFrame(data, columns=[" pose_Rx", " pose_Ry"])
    
    # ==================== CRITICAL: Z-SCORE NORMALIZATION ====================
    # Normalize using training data statistics (from config)
    df = (df - config['target_mean']) / config['target_std']
    
    # Apply amplification and exponential weighting (recent frames matter more)
    df = df * AMPLIFICATION_FACTOR
    weights = np.geomspace(0.05, 3.0, WINDOW_SIZE)
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


# ==================== MAIN PIPELINE ====================
def main():
    print("=" * 60)
    print("OpenFace + TCCT-Net Realtime Engagement Detection")
    print("(Terminal Mode - No Video Display)")
    print("=" * 60)
    
    # Load config and model
    config = load_config('TCCT_Net/config.json')
    model, device = load_model(config)
    print("[TCCT-Net] Model loaded successfully")
    
    # Initialize OpenFace extractor (owns camera device exclusively)
    openface = OpenFaceExtractor(OPENFACE_BIN)
    if not openface.available:
        print("[ERROR] OpenFace not available")
        return
    
    # Initialize sleep detector
    sleep_detector = SleepDetector()
    
    # Pose buffer for inference
    pose_buffer = deque(maxlen=WINDOW_SIZE)
    
    # Tracking
    frame_count = 0
    pred_count = 0
    last_result_time = time.time()
    last_prediction_time = time.time()
    buffer_initialized = False
    engagement_class = None
    
    print("\n[Pipeline] Reading OpenFace CSV results in real-time...")
    print("[Pipeline] Press Ctrl+C to stop\n")
    
    try:
        while True:
            time.sleep(0.5)  # Check every 500ms for new results
            
            # ==================== READ OPENFACE CSV ====================
            if os.path.exists(openface.csv_path):
                try:
                    df = pd.read_csv(openface.csv_path)
                    
                    if len(df) > 0:
                        # Filter to confident frames
                        if ' confidence' in df.columns:
                            df = df[df[' confidence'] >= 0.5]
                        
                        if len(df) > 0:
                            current_time = time.time()
                            
                            # Add poses to buffer
                            for idx, row in df.iterrows():
                                if ' pose_Rx' in row and ' pose_Ry' in row:
                                    try:
                                        pose_buffer.append([float(row[' pose_Rx']), float(row[' pose_Ry'])])
                                        frame_count += 1
                                    except (ValueError, TypeError):
                                        pass
                            
                            # ==================== SLEEP DETECTION ====================
                            is_sleeping = False
                            ear_val = None
                            
                            # Process latest frame through sleep detector
                            latest_row = df.iloc[-1]
                            is_sleeping, ear_val, _ = sleep_detector.update(latest_row)
                            
                            if ear_val is not None:
                                consec = sleep_detector.consecutive_closed_frames
                                threshold = sleep_detector.sleep_frame_threshold
                                print(f"[Sleep] EAR={ear_val:.4f} (threshold={sleep_detector.ear_threshold:.4f}), "
                                      f"Consecutive={consec}/{threshold}, Sleeping={is_sleeping}")
                            
                            # ==================== PREDICTIONS ====================
                            time_since_pred = current_time - last_prediction_time
                            min_time_between_preds = 1.0 / (30 / FRAMES_BETWEEN_PREDICTIONS)  # Approximate timing
                            
                            if len(pose_buffer) >= WINDOW_SIZE and time_since_pred >= min_time_between_preds:
                                try:
                                    if is_sleeping:
                                        engagement_class, status = get_final_engagement(None, is_sleeping=True)
                                        print(f"\n>>> [PREDICTION #{pred_count + 1}] Engagement: {status} (SLEEPING)")
                                    else:
                                        tcct_pred = predict_engagement(model, device, pose_buffer, config)
                                        engagement_class, status = get_final_engagement(tcct_pred, is_sleeping=False)
                                        label_text, _ = ENGAGEMENT_LABELS[engagement_class]
                                        print(f"\n>>> [PREDICTION #{pred_count + 1}] Engagement: {label_text} (class {engagement_class})")
                                    
                                    pred_count += 1
                                    last_prediction_time = current_time
                                    buffer_initialized = True
                                    
                                except Exception as e:
                                    print(f"[ERROR] Prediction failed: {e}")
                            
                            elif len(pose_buffer) < WINDOW_SIZE:
                                buffer_pct = int((len(pose_buffer) / WINDOW_SIZE) * 100)
                                if frame_count % 30 == 0:  # Print every ~30 frames
                                    print(f"[Warmup] Buffer: {buffer_pct}% ({len(pose_buffer)}/{WINDOW_SIZE} frames)")
                
                except (pd.errors.ParserError, pd.errors.EmptyDataError):
                    # CSV might be partially written
                    pass
                except Exception as e:
                    pass
    
    except KeyboardInterrupt:
        print("\n\n[Pipeline] Stopping...")
    
    finally:
        openface.cleanup()
        print("[Pipeline] Closed successfully")


if __name__ == "__main__":
    main()
