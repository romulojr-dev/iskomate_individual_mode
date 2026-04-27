import asyncio
import websockets
import sys
import os
import json
import torch
import time
import numpy as np
import pandas as pd
import subprocess
import threading
from collections import deque
from pathlib import Path
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'TCCT_Net'))

from TCCT_Net.models.feature_fusion import Decision_Fusion
from TCCT_Net.data.data_processing import batch_cwt

# ==================== CONFIG ====================
PROFILE = "custom"
WINDOW_SIZE = 280

if PROFILE == "fast":
    FRAMES_BETWEEN_PREDICTIONS = 5
elif PROFILE == "accurate":
    FRAMES_BETWEEN_PREDICTIONS = 10
elif PROFILE == "custom":
    FRAMES_BETWEEN_PREDICTIONS = 3
else:  # balanced
    FRAMES_BETWEEN_PREDICTIONS = 8

ENGAGEMENT_LABELS = {
    0: ("Not Engaged",    (0, 0, 255)),
    1: ("Barely Engaged", (0, 165, 255)),
    2: ("Engaged",        (0, 255, 255)),
    3: ("Highly Engaged", (0, 255, 0)),
}

AMPLIFICATION_FACTOR = 2.0
OPENFACE_BIN = r"C:\OpenFace\FeatureExtraction.exe"

# ==================== OPENFACE EXTRACTOR ====================
class OpenFaceExtractor:
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
            self._start_openface()
    
    def _start_openface(self):
        self.csv_path = os.path.join(self.output_dir, "face_out.csv")
        
        if os.path.exists(self.csv_path):
            try:
                os.remove(self.csv_path)
            except:
                pass
        
        cmd = [
            self.openface_bin,
            "-device", "0",
            "-out_dir", self.output_dir,
            "-of", "face_out",
            "-pose",
            "-2Dfp",
            "-3Dfp",
            "-aus",
        ]
        
        try:
            print("[OpenFace] Starting live stream processing (device owned by OpenFace)...")
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1
            )
            print("[OpenFace] Live stream subprocess started")
        except Exception as e:
            print(f"[ERROR] Failed to start OpenFace: {e}")
            self.available = False

    def cleanup(self):
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
    data = list(pose_buffer)
    while len(data) < WINDOW_SIZE:
        data.insert(0, data[0] if data else [0.0, 0.0])
    
    df = pd.DataFrame(data, columns=[" pose_Rx", " pose_Ry"])
    
    # Z-SCORE NORMALIZATION
    df = (df - config['target_mean']) / config['target_std']
    
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

# ==================== WEBSOCKET INTEGRATION ====================
class VerdictManager:
    def __init__(self):
        self.loop = None
        self.queue = None

    def set_loop(self, loop):
        self.loop = loop
        self.queue = asyncio.Queue()

    def put_verdict(self, verdict):
        if self.loop and self.queue:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, verdict)

verdict_manager = VerdictManager()

def run_ai_pipeline():
    print("=" * 60)
    print("OpenFace + TCCT-Net Realtime Engagement Detection (WebSocket Server)")
    print("(Terminal Mode - No Video Display)")
    print("=" * 60)
    
    config = load_config('TCCT_Net/config.json')
    model, device = load_model(config)
    print("[TCCT-Net] Model loaded successfully")
    
    openface = OpenFaceExtractor(OPENFACE_BIN)
    if not openface.available:
        print("[ERROR] OpenFace not available")
        os._exit(1)
        return
    
    pose_buffer = deque(maxlen=WINDOW_SIZE)
    
    frame_count = 0
    pred_count = 0
    last_prediction_time = time.time()
    
    print("\n[Pipeline] Reading OpenFace CSV results in real-time...")
    print("[Pipeline] Awaiting connections and running inference...\n")
    
    try:
        while True:
            time.sleep(0.5)
            
            if os.path.exists(openface.csv_path):
                try:
                    df = pd.read_csv(openface.csv_path)
                    
                    if len(df) > 0:
                        if ' confidence' in df.columns:
                            df = df[df[' confidence'] >= 0.5]
                        
                        if len(df) > 0:
                            current_time = time.time()
                            
                            for idx, row in df.iterrows():
                                if ' pose_Rx' in row and ' pose_Ry' in row:
                                    try:
                                        pose_buffer.append([float(row[' pose_Rx']), float(row[' pose_Ry'])])
                                        frame_count += 1
                                    except (ValueError, TypeError):
                                        pass
                            
                            time_since_pred = current_time - last_prediction_time
                            min_time_between_preds = 1.0 / (30 / FRAMES_BETWEEN_PREDICTIONS)
                            
                            if len(pose_buffer) >= WINDOW_SIZE and time_since_pred >= min_time_between_preds:
                                try:
                                    tcct_pred = predict_engagement(model, device, pose_buffer, config)
                                    label_text, _ = ENGAGEMENT_LABELS[tcct_pred]
                                    print(f"\n>>> [PREDICTION #{pred_count + 1}] Engagement: {label_text} (class {tcct_pred})")
                                    # Use the text label
                                    verdict_manager.put_verdict(label_text)
                                    
                                    pred_count += 1
                                    last_prediction_time = current_time
                                    
                                except Exception as e:
                                    print(f"[ERROR] Prediction failed: {e}")
                            
                            elif len(pose_buffer) < WINDOW_SIZE:
                                if frame_count % 30 == 0:
                                    buffer_pct = int((len(pose_buffer) / WINDOW_SIZE) * 100)
                                    print(f"[Warmup] Buffer: {buffer_pct}% ({len(pose_buffer)}/{WINDOW_SIZE} frames)")
                
                except (pd.errors.ParserError, pd.errors.EmptyDataError):
                    pass
                except Exception as e:
                    pass
                    
    except KeyboardInterrupt:
        print("\n\n[Pipeline] Stopping...")
    finally:
        openface.cleanup()
        print("[Pipeline] Closed successfully")
        os._exit(0)

async def stream_verdicts(websocket):
    print(f"\n[WebSocket] Client connected: {websocket.remote_address}")
    try:
        while True:
            verdict = await verdict_manager.queue.get()
            await websocket.send(verdict)
            print(f"[WebSocket] Sent: {verdict}")
    except websockets.exceptions.ConnectionClosed:
        print(f"\n[WebSocket] Client disconnected: {websocket.remote_address}")

async def start_server():
    verdict_manager.set_loop(asyncio.get_running_loop())
    
    ai_thread = threading.Thread(target=run_ai_pipeline, daemon=True)
    ai_thread.start()

    host = "0.0.0.0"
    port = 8765
    print(f"\n[WebSocket Server] Starting on ws://{host}:{port}")
    print("[WebSocket Server] Awaiting Flutter client connection...")
    
    async with websockets.serve(stream_verdicts, host, port):
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
        os._exit(0)
