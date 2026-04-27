"""
OpenFace Setup & Configuration Guide
====================================

This guide helps you configure OpenFace for the realtime pipeline.

OPTION 1: Standalone OpenFace (Windows) - RECOMMENDED
=====================================================

1. Download OpenFace pre-built binaries from:
   https://github.com/TadasBaltrusaitis/OpenFace/releases
   
   Download the Windows version: OpenFace_2.2.0_win_x64.zip

2. Extract to a known location, e.g.:
   C:\OpenFace\

3. Inside, you should find:
   C:\OpenFace\FeatureExtraction.exe
   
4. Update realtime_engagement_openface.py line 35:
   OPENFACE_BIN = r"C:\OpenFace\FeatureExtraction.exe"

5. Test OpenFace manually:
   > C:\OpenFace\FeatureExtraction.exe -help
   
   You should see a list of available options.

6. Run the pipeline:
   python realtime_engagement_openface.py


OPTION 2: Build from Source (Linux/Mac)
========================================

Follow official guide: https://github.com/TadasBaltrusaitis/OpenFace/wiki/Unix-Installation

After build, update OPENFACE_BIN to:
   OPENFACE_BIN = "/path/to/OpenFace/build/bin/FeatureExtraction"


OPTION 3: Docker (All Platforms)
==================================

If you have Docker, run OpenFace in a container:

    docker run -it -v $(pwd):/workspace \
        algebr/openface:latest \
        FeatureExtraction -f /workspace/input_video.mp4 \
                          -out_dir /workspace/output

Then update realtime_engagement_openface.py to call the Docker container.


OPENFACE FEATURE EXTRACTION PARAMETERS
======================================

The pipeline uses these OpenFace flags:

    -f <file>          : input video file
    -out_dir <dir>     : output directory for CSV
    -2Dfp               : output 2D face landmarks
    -3Dfp               : output 3D face landmarks
    -pose               : output head pose (pose_Rx, pose_Ry, pose_Rz)
    -au                 : output action units

Output CSV contains:
    - frame_num: frame number
    -  pose_Rx: head rotation X (pitch) in radians
    - pose_Ry: head rotation Y (yaw) in radians
    - pose_Rz: head rotation Z (roll) in radians
    - ... other features (AUs, landmarks, etc.)


TCCT-NET EXPECTS ONLY:
    pose_Rx and pose_Ry (head pitch and yaw)


PIPELINE FLOW
=============

1. Capture frames from webcam
   └─ buffer into `frame_buffer` (CHUNK_SIZE=30 frames)

2. When buffer is full:
   └─ write frames to temporary .mp4 file
   
3. Run OpenFace FeatureExtraction on temp .mp4
   └─ generates CSV with pose features
   
4. Read pose_Rx and pose_Ry from CSV
   └─ append to `pose_buffer` (WINDOW_SIZE=280)
   
5. When pose_buffer is full and frame count % 10 == 0:
   └─ run TCCT-Net inference
   └─ predict engagement class (0-3)
   
6. Display prediction on screen
   └─ update every ~300ms


TROUBLESHOOTING
===============

Q: "OpenFace binary not found"
A: Update OPENFACE_BIN to correct path. Run:
   python -c "import subprocess; subprocess.run(['path/to/FeatureExtraction.exe', '-help'])"

Q: OpenFace runs but no CSV output
A: Check that output directory is writable. Verify with:
   python -c "import os; os.makedirs('test_dir', exist_ok=True)"

Q: Predictions are always the same or wrong
A: Check that TCCT_Net/config.json exists and has correct parameters:
   - freq_min, freq_max, tensor_height, sampling_frequency

Q: Pipeline is very slow
A: This is expected because OpenFace runs every CHUNK_SIZE frames.
   Trade-off: accuracy (OpenFace correct features) vs. latency.
   To speed up:
   - Reduce CHUNK_SIZE (e.g., 15 instead of 30)
   - Reduce WINDOW_SIZE (e.g., 140 instead of 280)
   - Run on GPU if available


PERFORMANCE EXPECTATIONS
=========================

Typical latency per prediction:
    - Frame capture (30 frames):       ~1 second
    - OpenFace extraction:              3-5 seconds (depends on CPU)
    - TCCT-Net inference:               <100ms
    - Total per prediction cycle:       5-7 seconds

Engagement class updates: ~every 7-10 seconds (depending on prediction frequency)


COMPARISON: OpenFace vs MediaPipe
==================================

                OpenFace       MediaPipe
                --------       ---------
Domain Match    ✓ YES          ✗ NO (different features)
Accuracy        ✓ Higher       - Lower
Speed           ✗ Slower       ✓ Faster
Realtime        ~ Partial      ✓ Full realtime
Setup           ✗ Complex      ✓ Easy
CSV dependency  ✓ Yes (batch)  ✗ No (frame-by-frame)

Recommendation: Use OpenFace for production (correct domain).
                Use MediaPipe for prototyping/testing (faster).
"""

import os
import subprocess
import platform

def verify_openface(openface_bin):
    """Verify OpenFace is installed and working."""
    print("Verifying OpenFace installation...")
    
    if not os.path.exists(openface_bin):
        print(f"❌ OpenFace not found at: {openface_bin}")
        print("   Please download and install from:")
        print("   https://github.com/TadasBaltrusaitis/OpenFace/releases")
        return False
    
    try:
        result = subprocess.run(
            [openface_bin, "-help"],
            capture_output=True,
            timeout=5,
            text=True
        )
        if result.returncode == 0 or "Usage" in result.stdout or "Usage" in result.stderr:
            print(f"✓ OpenFace found: {openface_bin}")
            return True
    except Exception as e:
        print(f"❌ Error running OpenFace: {e}")
    
    return False


def verify_tcct_net():
    """Verify TCCT-Net model and config are available."""
    print("Verifying TCCT-Net setup...")
    
    config_path = "TCCT_Net/config.json"
    model_path = "TCCT_Net/final_model_weights.pth"
    
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        return False
    print(f"✓ Config found: {config_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model weights not found: {model_path}")
        return False
    print(f"✓ Model weights found: {model_path}")
    
    return True


def main():
    print("\n" + "="*60)
    print("OpenFace + TCCT-Net Setup Verification")
    print("="*60 + "\n")
    
    # Detect platform
    system = platform.system()
    print(f"System: {system}")
    
    # Suggest OpenFace path based on OS
    if system == "Windows":
        suggested_path = r"C:\OpenFace\FeatureExtraction.exe"
    elif system == "Darwin":
        suggested_path = "/usr/local/bin/FeatureExtraction"
    else:
        suggested_path = "/usr/bin/FeatureExtraction"
    
    print(f"Suggested OpenFace path: {suggested_path}\n")
    
    # Verify
    of_ok = verify_openface(suggested_path)
    tc_ok = verify_tcct_net()
    
    print("\n" + "="*60)
    if of_ok and tc_ok:
        print("✓ All checks passed! Ready to run realtime_engagement_openface.py")
    else:
        print("❌ Some checks failed. See above for details.")
    print("="*60 + "\n")
    
    return of_ok and tc_ok


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
