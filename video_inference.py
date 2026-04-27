"""
Video-to-Inference Pipeline: OpenFace + TCCT-Net
=================================================

Process a pre-recorded video file with engagement detection:
1. Extract frames from video
2. Run OpenFace FeatureExtraction on the entire video
3. Chunk pose data into 280-frame segments
4. Run TCCT-Net inference on each segment
5. Display predictions with frame ranges

Usage:
    python video_inference.py <video_file> [--profile balanced]
    
Example:
    python video_inference.py test_video.mp4
    python video_inference.py student_recording.mov --profile accurate
"""

import cv2
import sys
import os
import json
import torch
import argparse
import tempfile
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from collections import deque
from torch import nn

# Add TCCT_Net to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'TCCT_Net'))

from TCCT_Net.models.feature_fusion import Decision_Fusion
from TCCT_Net.data.data_processing import batch_cwt


# ==================== CONFIG ====================
WINDOW_SIZE = 280
AMPLIFICATION_FACTOR = 2.0

ENGAGEMENT_LABELS = {
    0: ("Not Engaged",    (0, 0, 255)),
    1: ("Barely Engaged", (0, 165, 255)),
    2: ("Engaged",        (0, 255, 255)),
    3: ("Highly Engaged", (0, 255, 0)),
}

OPENFACE_BIN = r"C:\OpenFace\FeatureExtraction.exe"


# ==================== OPENFACE EXTRACTOR ====================
class OpenFaceExtractor:
    """Extract head pose features from video using OpenFace."""
    
    def __init__(self, openface_bin=OPENFACE_BIN, fps=30):
        self.openface_bin = openface_bin
        self.fps = fps
        self.temp_dir = tempfile.mkdtemp(prefix="openface_")
        
        if not os.path.exists(self.openface_bin):
            raise FileNotFoundError(f"OpenFace binary not found at: {self.openface_bin}")
    
    def extract_from_video(self, video_path):
        """
        Extract pose features from entire video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            DataFrame with columns [' pose_Rx', ' pose_Ry'] or None if failed
        """
        try:
            # Run OpenFace FeatureExtraction on video
            csv_path = self._run_openface(video_path)
            
            if csv_path and os.path.exists(csv_path):
                # Read and return pose data
                df = pd.read_csv(csv_path)
                
                # OpenFace outputs pose_Rx and pose_Ry (in radians)
                if ' pose_Rx' in df.columns and ' pose_Ry' in df.columns:
                    result = df[[' pose_Rx', ' pose_Ry']].copy()
                    return result
        except Exception as e:
            print(f"[OpenFace Error] {e}")
        
        return None
    
    def _run_openface(self, video_path):
        """
        Run OpenFace FeatureExtraction on video.
        Returns path to output CSV if successful.
        """
        output_dir = r"C:\OpenFace\output_au"
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = [
            self.openface_bin,
            "-f", video_path,
            "-out_dir", output_dir,
            "-device", "0",
            "-2Dfp",  # output 2D face landmarks
            "-3Dfp",  # output 3D face landmarks
            "-pose",  # output head pose
            "-aus",   # output action units
        ]
        
        try:
            print(f"[OpenFace] Processing video: {video_path}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=300,  # 5 minute timeout
                text=True
            )
            
            if result.returncode != 0:
                print(f"[OpenFace] Error: {result.stderr}")
                return None
            
            # Find generated CSV file
            csv_files = list(Path(output_dir).glob("*.csv"))
            if csv_files:
                csv_path = str(csv_files[0])
                print(f"[OpenFace] ✓ Generated CSV: {csv_path}")
                return csv_path
        except subprocess.TimeoutExpired:
            print("[OpenFace] Timeout (>5min)")
        except Exception as e:
            print(f"[OpenFace] Subprocess error: {e}")
        
        return None
    
    def cleanup(self):
        """Remove temporary directory."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


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
        pose_buffer: deque or list of [pose_Rx, pose_Ry] pairs
        config: config dict with target_mean and target_std for normalization
        
    Returns:
        predicted engagement class (0-3) and output logits
    """
    data = list(pose_buffer)
    
    # Pad with first sample if not enough data
    while len(data) < WINDOW_SIZE:
        data.insert(0, data[0] if data else [0.0, 0.0])
    
    # Keep only first WINDOW_SIZE samples
    data = data[:WINDOW_SIZE]
    
    df = pd.DataFrame(data, columns=[" pose_Rx", " pose_Ry"])
    
    # ==================== CRITICAL: Z-SCORE NORMALIZATION ====================
    # This is essential for model accuracy! Uses training data statistics.
    df = (df - config['target_mean']) / config['target_std']
    
    # Apply amplification and weighting (same as training)
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
        logits = output[0].cpu().numpy()
    
    return predicted_class, logits


# ==================== VIDEO PROCESSING ====================
def get_video_fps(video_path):
    """Get FPS from video file."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30


def process_video_file(video_path, output_csv=None, step_size=None):
    """
    Process video file and extract engagement predictions using sliding window.
    
    Args:
        video_path: Path to video file
        output_csv: Optional path to save results CSV
        step_size: Optional sliding window step (default: 20 frames, ~0.67s at 30fps)
                   Smaller = more predictions but slower
                   Larger = fewer predictions but faster
        
    Returns:
        List of predictions with frame ranges
    """
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return None
    
    # Default step size: 20 frames (~0.67 seconds at 30 fps)
    # This means: 279 frames reused + 1 new frame = new prediction
    if step_size is None:
        step_size = 20
    
    print("=" * 70)
    print(f"Video Inference Pipeline: {video_path}")
    print(f"Sliding Window: 280 frames, Step: {step_size} frames")
    print("=" * 70)
    
    # Load config and model
    config = load_config('TCCT_Net/config.json')
    model, device = load_model(config)
    print("[TCCT-Net] Model loaded successfully\n")
    
    # Initialize OpenFace extractor
    openface = OpenFaceExtractor(OPENFACE_BIN)
    
    try:
        # Step 1: Get video info
        video_fps = get_video_fps(video_path)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        print(f"[Video] FPS: {video_fps:.1f}, Total Frames: {total_frames}")
        print(f"[Video] Duration: {total_frames / video_fps:.1f}s\n")
        
        # Step 2: Extract pose features with OpenFace
        print("[Step 1/3] Extracting OpenFace features...")
        pose_df = openface.extract_from_video(video_path)
        
        if pose_df is None or len(pose_df) == 0:
            print("[ERROR] Failed to extract pose features")
            return None
        
        print(f"[OpenFace] ✓ Extracted {len(pose_df)} pose samples\n")
        
        # Step 3: Create sliding windows
        print(f"[Step 2/3] Creating sliding windows (step size: {step_size} frames)...")
        windows = []
        
        # Start from 0 and step by step_size
        for start_idx in range(0, len(pose_df) - WINDOW_SIZE + 1, step_size):
            end_idx = start_idx + WINDOW_SIZE
            chunk_data = pose_df.iloc[start_idx:end_idx]
            
            if len(chunk_data) >= WINDOW_SIZE:  # Only process complete windows
                windows.append({
                    'start_frame': start_idx,
                    'end_frame': end_idx,
                    'data': chunk_data
                })
        
        print(f"[Sliding Windows] ✓ Created {len(windows)} overlapping windows\n")
        
        # Step 4: Run inference on each window
        print("[Step 3/3] Running inference on each window...\n")
        print("-" * 70)
        print(f"{'Window':<10} {'Frame Range':<20} {'Prediction':<20} {'Confidence':<15}")
        print("-" * 70)
        
        results = []
        
        for window_idx, window in enumerate(windows):
            start_frame = window['start_frame']
            end_frame = window['end_frame']
            chunk_data = window['data']
            
            # Convert to list of [pose_Rx, pose_Ry] pairs
            pose_buffer = [
                [row[' pose_Rx'], row[' pose_Ry']]
                for _, row in chunk_data.iterrows()
            ]
            
            # Run inference
            pred_class, logits = predict_engagement(model, device, pose_buffer, config)
            pred_label, _ = ENGAGEMENT_LABELS[pred_class]
            
            # Calculate confidence (softmax)
            confidence = np.exp(logits) / np.sum(np.exp(logits))
            confidence_pct = confidence[pred_class] * 100
            
            # Calculate time ranges
            start_time = start_frame / video_fps
            end_time = end_frame / video_fps
            
            result = {
                'window': window_idx + 1,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_time': start_time,
                'end_time': end_time,
                'prediction': pred_label,
                'predicted_class': pred_class,
                'confidence': confidence_pct,
                'logits': logits.tolist()
            }
            results.append(result)
            
            # Print result
            frame_range = f"{start_frame}-{end_frame}"
            time_range = f"({start_time:.1f}s-{end_time:.1f}s)"
            print(f"{window_idx+1:<10} {frame_range:<10} {time_range:<10} {pred_label:<20} {confidence_pct:.1f}%")
        
        print("-" * 70)
        
        # Step 5: Save results if requested
        if output_csv:
            results_df = pd.DataFrame([
                {
                    'window': r['window'],
                    'start_frame': r['start_frame'],
                    'end_frame': r['end_frame'],
                    'start_time_s': r['start_time'],
                    'end_time_s': r['end_time'],
                    'prediction': r['prediction'],
                    'confidence_%': r['confidence']
                }
                for r in results
            ])
            results_df.to_csv(output_csv, index=False)
            print(f"\n[Results] ✓ Saved to: {output_csv}")
        
        # Summary statistics
        print(f"\n[Summary]")
        print(f"  Total windows: {len(results)}")
        print(f"  Predictions per minute: {len(results) / (total_frames / video_fps / 60):.1f}")
        predictions_count = {}
        for r in results:
            label = r['prediction']
            predictions_count[label] = predictions_count.get(label, 0) + 1
        
        for label, count in predictions_count.items():
            pct = (count / len(results)) * 100
            print(f"  {label}: {count} windows ({pct:.1f}%)")
        
        return results
        
    finally:
        openface.cleanup()
        print(f"\n[Pipeline] ✓ Completed successfully")


# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser(
        description="Process pre-recorded video with TCCT-Net engagement detection using sliding window",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sliding Window Modes:

  Default (step=20): 
    - 280-frame window, step by 20 frames
    - ~0.67s between predictions at 30fps
    - Good balance between accuracy and speed
  
  Fast (step=50):
    - Fewer predictions, ~10 per minute
    - Faster processing, lower accuracy
  
  Detailed (step=5):
    - More predictions, catches transitions
    - Slower processing, more responsive

Examples:
  python video_inference.py test_video.mp4
  python video_inference.py student.mov --output results.csv
  python video_inference.py long_class.mp4 --output predictions.csv --step 50
  python video_inference.py detailed_analysis.mp4 --step 5
        """
    )
    
    parser.add_argument(
        'video',
        help='Path to video file (mp4, mov, avi, etc.)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Optional: Save results to CSV file'
    )
    
    parser.add_argument(
        '--step',
        type=int,
        default=20,
        help='Sliding window step size in frames (default: 20, ~0.67s at 30fps)'
    )
    
    args = parser.parse_args()
    
    # Check if video exists
    if not os.path.exists(args.video):
        print(f"[ERROR] Video file not found: {args.video}")
        sys.exit(1)
    
    # Generate output filename if not provided
    output_csv = args.output
    if output_csv is None:
        video_name = Path(args.video).stem
        output_csv = f"{video_name}_predictions.csv"
    
    # Process video with sliding window
    results = process_video_file(args.video, output_csv, step_size=args.step)
    
    if results is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
