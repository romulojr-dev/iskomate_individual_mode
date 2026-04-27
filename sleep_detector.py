import numpy as np
import pandas as pd
from collections import deque

EAR_THRESHOLD         = 0.4920   # midpoint between awake Q1 and sleeping mean
AU45_THRESHOLD        = 999.0    # effectively disabled — AU45 unreliable for sleep
SLEEP_FRAME_THRESHOLD = 90       # 3 seconds at 30fps — more robust than 60

# ── Correct OpenFace Landmark Indices ─────────────────────────────────────────
# Eye landmark indices in OpenFace output (using eye_lmk_x/y columns)
# Left eye:  outer(8), top-outer(1), top-center(2), top-inner(3),
#            inner(14), bot-inner(5), bot-center(6), bot-outer(7)
# Right eye: same pattern + 28 offset
LEFT_EYE_IDX  = [8, 2, 3, 14, 5, 6]
RIGHT_EYE_IDX = [36, 30, 31, 42, 33, 34]

def compute_ear_from_openface(row):
    """
    Compute Eye Aspect Ratio using OpenFace dedicated eye landmarks (eye_lmk_x/y columns).
    Uses the same computation as calibrate_sleep.py for consistency.
    """
    try:
        # Left eye
        lx = [float(row[f' eye_lmk_x_{i}']) for i in LEFT_EYE_IDX]
        ly = [float(row[f' eye_lmk_y_{i}']) for i in LEFT_EYE_IDX]

        # Right eye
        rx = [float(row[f' eye_lmk_x_{i}']) for i in RIGHT_EYE_IDX]
        ry = [float(row[f' eye_lmk_y_{i}']) for i in RIGHT_EYE_IDX]

        def ear(x, y):
            # EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
            v1 = np.sqrt((x[1]-x[5])**2 + (y[1]-y[5])**2)
            v2 = np.sqrt((x[2]-x[4])**2 + (y[2]-y[4])**2)
            h  = np.sqrt((x[0]-x[3])**2 + (y[0]-y[3])**2)
            return (v1 + v2) / (2.0 * h) if h > 0 else 0.0

        left_ear  = ear(lx, ly)
        right_ear = ear(rx, ry)
        return (left_ear + right_ear) / 2.0

    except Exception as e:
        return 0.3

class SleepDetector:
    def __init__(self,
                 ear_threshold=EAR_THRESHOLD,
                 au45_threshold=AU45_THRESHOLD,
                 sleep_frame_threshold=SLEEP_FRAME_THRESHOLD):

        self.ear_threshold        = ear_threshold
        self.au45_threshold       = au45_threshold
        self.sleep_frame_threshold = sleep_frame_threshold

        self.consecutive_closed_frames = 0
        self.is_sleeping              = False
        self.ear_buffer               = deque(maxlen=90)  # 3 second history
        self.au45_buffer              = deque(maxlen=90)

    def update(self, row):
        """
        Simple eye closure detection.
        If eyes closed for 3 seconds (90 frames at 30fps) → sleeping.
        Returns (is_sleeping, ear_value, au45_value)
        """
        try:
            confidence = float(row.get(' confidence', 1.0))
            success    = int(row.get(' success', 1))
            if confidence < 0.5 or success == 0:
                return self.is_sleeping, None, None
        except Exception:
            pass

        # Compute EAR
        ear = compute_ear_from_openface(row)

        # Get AU45
        try:
            au45 = float(row.get(' AU45_r', 0.0))
        except Exception:
            au45 = 0.0

        self.ear_buffer.append(ear)
        self.au45_buffer.append(au45)

        # Simple logic: eye_closed if EAR below threshold
        eye_closed = (ear < self.ear_threshold)

        if eye_closed:
            self.consecutive_closed_frames += 1
        else:
            # Instant reset when eyes open
            self.consecutive_closed_frames = 0

        # Determine sleep state: sleeping if closed for 3 seconds (90 frames)
        if self.consecutive_closed_frames >= self.sleep_frame_threshold:
            self.is_sleeping = True
        else:
            self.is_sleeping = False

        return self.is_sleeping, ear, au45

    def get_perclos(self):
        """
        Compute PERCLOS over the current buffer (percentage of time eyes closed).
        Useful for thesis reporting.
        """
        if len(self.ear_buffer) == 0:
            return 0.0
        closed_frames = sum(1 for e in self.ear_buffer if e < self.ear_threshold)
        return closed_frames / len(self.ear_buffer)

    def reset(self):
        self.consecutive_closed_frames = 0
        self.is_sleeping = False
        self.ear_buffer.clear()
        self.au45_buffer.clear()


def get_final_engagement(tcct_prediction, is_sleeping):
    """
    Final decision layer that fuses TCCT-Net output with sleep detection.
    Sleep overrides engagement classification.
    
    Logic:
    1. If sleeping → return "Not Engaged" (class 0)
    2. Else → use TCCT-Net verdict (class 0-3)
    """
    if is_sleeping:
        return 0, "Not Engaged (Sleeping)"  # Sleeping = Not Engaged

    labels = {
        0: "Not Engaged",
        1: "Barely Engaged",
        2: "Engaged",
        3: "Highly Engaged"
    }
    return tcct_prediction, labels.get(tcct_prediction, "Unknown")