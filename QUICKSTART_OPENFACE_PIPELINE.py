"""
QUICK START: OpenFace + TCCT-Net Realtime Pipeline
==================================================

STEP 1: Verify Setup (One-time)
-------------------------------

    python openface_setup.py

This will check:
    ✓ OpenFace is installed
    ✓ TCCT-Net config and weights exist
    ✓ All dependencies are ready

If any check fails, follow the instructions in openface_setup.py


STEP 2: Configure OpenFace Path
-------------------------------

Open realtime_engagement_openface.py and update line 35:

    OPENFACE_BIN = r"C:\OpenFace\FeatureExtraction.exe"  # Windows
    # or
    OPENFACE_BIN = "/usr/local/bin/FeatureExtraction"    # macOS
    # or
    OPENFACE_BIN = "/usr/bin/FeatureExtraction"          # Linux


STEP 3: Adjust Pipeline Parameters (Optional)
---------------------------------------------

In realtime_engagement_openface.py, you can tune:

    CHUNK_SIZE = 30                    # frames between OpenFace runs
                                       # ↑ larger = fewer OpenFace calls, but higher latency
                                       # ↓ smaller = more OpenFace calls, but more CPU

    FRAMES_BETWEEN_PREDICTIONS = 10    # how often to run TCCT-Net inference
                                       # ↑ larger = less frequent predictions
                                       # ↓ smaller = more frequent predictions

    AMPLIFICATION_FACTOR = 2.0         # pose signal amplification (same as original)

    WINDOW_SIZE = 280                  # IMPORTANT: must match TCCT-Net training config
                                       # Check: TCCT_Net/config.json


STEP 4: Run the Pipeline
-------------------------------

Activate virtual environment (if needed):

    .\engagement_env\Scripts\Activate.ps1  # Windows

Run the pipeline:

    python realtime_engagement_openface.py

You should see:

    ============================================================
    OpenFace + TCCT-Net Realtime Engagement Detection
    ============================================================
    [TCCT-Net] Model loaded successfully
    [OpenFace] Using temp dir: C:\Users\...\AppData\Local\Temp\openface_...
    [Camera] Opened successfully
    [Pipeline] Starting... Press 'q' to quit

    [Frame 30] Processing 30 frames with OpenFace...
    [OpenFace] Wrote 30 frames to ...
    [OpenFace] Running: C:\OpenFace\FeatureExtraction.exe -f ...
    [OpenFace] Generated CSV: ...
    [Prediction #1] Engagement: Engaged


STEP 5: Interpret Output
-------------------------------

On the video window, you'll see:

    ┌────────────────────────────────────────┐
    │ Pose Buffer: 45%                  FPS: 28.5
    │ Frame Chunk: 15/30                      │
    │ ┌──────────────────────────────────┐   │
    │ │ Engagement: Engaged              │   │ ← Green = Highly Engaged
    │ │                                  │   │ ← Yellow = Engaged
    │ └──────────────────────────────────┘   │ ← Orange = Barely Engaged
    │                                         │ ← Red = Not Engaged
    │                                         │
    │                                         │
    └────────────────────────────────────────┘

Labels:
    Not Engaged     (0) → Red box
    Barely Engaged  (1) → Orange box
    Engaged         (2) → Yellow box
    Highly Engaged  (3) → Green box


STEP 6: Stop the Pipeline
-------------------------------

Press 'q' on the video window, or close the window.

The script will:
    - Close the camera
    - Clean up temporary files
    - Exit gracefully


UNDERSTANDING THE LATENCY
==========================

Total latency breakdown (approximate):

    30 frames at 30 FPS           =  1.0 second (capture)
  + OpenFace processing           =  3-5 seconds (CPU-dependent)
  + TCCT-Net inference            =  0.1 seconds
  ─────────────────────────────────────────────────
  Total per prediction cycle      =  4-6 seconds

This means engagement predictions update every ~5 seconds.

If this is too slow:
    1. Reduce CHUNK_SIZE (e.g., 15 instead of 30)
    2. Use a faster CPU / GPU acceleration for OpenFace
    3. Run OpenFace in parallel threads (advanced)


DEBUGGING TIPS
==============

Q: Pipeline hangs after "Processing ... frames with OpenFace"
A: OpenFace is slow or not responding. Check:
   1. Is OPENFACE_BIN path correct?
   2. Can you run OpenFace manually from command line?
   3. Do you have temp directory write permissions?

Q: OpenFace generates no CSV
A: Check permissions and output directory:
   1. Verify output directory is writable
   2. Check Windows Defender / antivirus isn't blocking OpenFace

Q: Model predictions don't change
A: Possible causes:
   1. Head is too still (not enough pose variation)
   2. TCCT-Net config mismatch (check WINDOW_SIZE, freq_min, freq_max)
   3. OpenFace pose values are not being extracted correctly

Q: Camera won't open
A: 
   1. Make sure no other program is using camera
   2. Check camera permissions (Settings > Privacy > Camera)
   3. Try unplugging/replugging USB camera


COMPARING WITH ORIGINAL MEDIAPIPE VERSION
===========================================

Original (mediapipe_extractor.py):
    ✓ Fast, realtime (frame-by-frame)
    ✗ Domain mismatch with TCCT-Net training
    ✗ Poor engagement predictions

New (realtime_engagement_openface.py):
    ✓ Uses same feature domain as training (OpenFace)
    ✓ Better engagement accuracy
    ✗ Slower latency (~5 seconds per update)
    ✗ Requires OpenFace installation


NEXT STEPS
==========

1. Test the pipeline with webcam
2. Collect engagement ground-truth labels for validation
3. Fine-tune thresholds if engagement classes are skewed
4. Consider running OpenFace in GPU mode for speed
5. For production: add error recovery and logging
"""

if __name__ == "__main__":
    print(__doc__)
