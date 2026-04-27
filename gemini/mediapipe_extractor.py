import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math
import time
from collections import deque

FACE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 330.0, -65.0],
    [-225.0, -170.0, -135.0],
    [225.0, -170.0, -135.0],
    [-150.0, 150.0, -125.0],
    [150.0, 150.0, -125.0]
], dtype=np.float64)

LANDMARK_IDS = [1, 152, 33, 263, 61, 291]
WINDOW_SIZE = 280

class HeadPoseExtractor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.2
        )
        self.pose_buffer = deque(maxlen=WINDOW_SIZE)

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
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rot_vec, _ = cv2.solvePnP(
            FACE_3D_MODEL, image_points, cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return None, None, None, frame

        rot_mat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
        pitch, yaw, roll = angles

        pitch_rad = math.radians(pitch)
        yaw_rad = math.radians(yaw)
        roll_rad = math.radians(roll)

        cv2.putText(frame, f"Pitch: {pitch_rad:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Yaw:   {yaw_rad:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Roll:  {roll_rad:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return pitch_rad, yaw_rad, roll_rad, frame

    def update_buffer(self, pitch, yaw):
        self.pose_buffer.append([pitch, yaw])

    def get_sequence_df(self):
        if len(self.pose_buffer) < WINDOW_SIZE:
            data = list(self.pose_buffer)
            while len(data) < WINDOW_SIZE:
                data.insert(0, data[0] if data else [0.0, 0.0])
        else:
            data = list(self.pose_buffer)
        return pd.DataFrame(data, columns=[" pose_Rx", " pose_Ry"])

    def release(self):
        self.face_mesh.close()

if __name__ == "__main__":
    try:
        extractor = HeadPoseExtractor()
        cap = cv2.VideoCapture(0)
        prev_time = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            pitch, yaw, roll, annotated = extractor.get_head_pose(frame)
            curr_time = time.time()

            if pitch is not None and (curr_time - prev_time) >= 1.0 / 30.0:
                extractor.update_buffer(pitch, yaw)
                prev_time = curr_time
                cv2.putText(annotated, f"Buffer: {len(extractor.pose_buffer)}/{WINDOW_SIZE}",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Head Pose Extraction", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        if len(extractor.pose_buffer) >= 10:
            df = extractor.get_sequence_df()
            df.to_csv("sample_pose_output.csv", index=False)
        cap.release()
        cv2.destroyAllWindows()
        extractor.release()