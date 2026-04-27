import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from collections import deque

WINDOW_SIZE = 280
SMOOTHING = 0.85

# MediaPipe canonical face model points (built-in 3D model)
FACE_3D_MODEL = np.array([
    [0.0,      0.0,      0.0    ],  # Nose tip (1)
    [0.0,     -63.6,    -12.5   ],  # Chin (152)  
    [-43.3,    32.7,    -26.0   ],  # Left eye corner (33)
    [43.3,     32.7,    -26.0   ],  # Right eye corner (263)
    [-28.9,   -28.9,    -24.1   ],  # Left mouth corner (61)
    [28.9,    -28.9,    -24.1   ]   # Right mouth corner (291)
], dtype=np.float64)

LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

class HeadPoseExtractor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.9
        )
        self.pose_buffer = deque(maxlen=WINDOW_SIZE)
        self.last_pitch = None
        self.last_yaw = None

    def get_head_pose(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None, None, None, frame

        landmarks = results.multi_face_landmarks[0].landmark

        image_points = np.array([
            [landmarks[idx].x * w, landmarks[idx].y * h]
            for idx in LANDMARK_IDS
        ], dtype=np.float64)

        focal_length = w
        cam_matrix = np.array([
            [focal_length, 0,            w / 2],
            [0,            focal_length, h / 2],
            [0,            0,            1    ]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rot_vec, _ = cv2.solvePnP(
            FACE_3D_MODEL,
            image_points,
            cam_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None, None, None, frame

        # Convert to rotation matrix
        rot_mat, _ = cv2.Rodrigues(rot_vec)

        # Extract Euler angles directly from rotation matrix
        # This is the standard aerospace/robotics convention
        pitch = np.arcsin(-rot_mat[2, 0])
        yaw   = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
        roll  = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])

        # Print raw values before any adjustment
        print(
            f"RAW -> Pitch: {np.degrees(pitch):+6.1f}  "
            f"Yaw: {np.degrees(yaw):+6.1f}  "
            f"Roll: {np.degrees(roll):+6.1f}",
            end='\r'
        )

        # Display on frame (no sign adjustment yet - we observe first)
        cv2.putText(frame, f"Pitch: {np.degrees(pitch):+.1f}deg", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Yaw:   {np.degrees(yaw):+.1f}deg",   (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Roll:  {np.degrees(roll):+.1f}deg",  (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return pitch, yaw, roll, frame

    def update_buffer(self, pitch, yaw):
        if self.last_pitch is None:
            self.last_pitch = pitch
            self.last_yaw   = yaw
        else:
            self.last_pitch = SMOOTHING * self.last_pitch + (1 - SMOOTHING) * pitch
            self.last_yaw   = SMOOTHING * self.last_yaw   + (1 - SMOOTHING) * yaw
        self.pose_buffer.append([self.last_pitch, self.last_yaw])

    def get_sequence_df(self):
        data = list(self.pose_buffer)
        while len(data) < WINDOW_SIZE:
            data.insert(0, data[0] if data else [0.0, 0.0])
        return pd.DataFrame(data, columns=[" pose_Rx", " pose_Ry"])

    def release(self):
        self.face_mesh.close()


if __name__ == "__main__":
    extractor = HeadPoseExtractor()
    cap = cv2.VideoCapture(0)

    print("Monitoring head pose. Press 'q' to quit.")
    print("Expected behavior:")
    print("  Looking straight -> pitch ~0, yaw ~0")
    print("  Tilt down        -> pitch NEGATIVE")
    print("  Turn left        -> yaw NEGATIVE")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pitch, yaw, roll, annotated = extractor.get_head_pose(frame)

        if pitch is not None:
            extractor.update_buffer(pitch, yaw)
            smoothed_pitch = extractor.last_pitch
            smoothed_yaw   = extractor.last_yaw
            print(
                f"Raw  -> Pitch: {np.degrees(pitch):+6.1f}  Yaw: {np.degrees(yaw):+6.1f} | "
                f"Smoothed -> Pitch: {np.degrees(smoothed_pitch):+6.1f}  Yaw: {np.degrees(smoothed_yaw):+6.1f}",
                end='\r'
            )

        cv2.imshow("Head Pose Extraction", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.release()