import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from collections import deque

WINDOW_SIZE = 280
SMOOTHING = 0.85

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

        # --- Use specific stable landmarks for pose estimation ---
        # These are well-distributed, stable points on the face
        nose_tip    = np.array([landmarks[4].x * w,   landmarks[4].y * h])
        chin        = np.array([landmarks[152].x * w,  landmarks[152].y * h])
        left_eye    = np.array([landmarks[33].x * w,   landmarks[33].y * h])
        right_eye   = np.array([landmarks[263].x * w,  landmarks[263].y * h])
        left_mouth  = np.array([landmarks[61].x * w,   landmarks[61].y * h])
        right_mouth = np.array([landmarks[291].x * w,  landmarks[291].y * h])

        # --- Compute pitch (vertical head tilt) ---
        # Ratio of vertical face span to horizontal span
        face_height = np.linalg.norm(chin - nose_tip)
        face_width  = np.linalg.norm(right_eye - left_eye)
        eye_center  = (left_eye + right_eye) / 2

        # Pitch: angle between eye center and chin relative to vertical
        vertical_vec = chin - eye_center
        pitch = np.arctan2(vertical_vec[0], vertical_vec[1])  # radians

        # Yaw: asymmetry between left and right facial landmarks
        left_dist  = np.linalg.norm(nose_tip - left_eye)
        right_dist = np.linalg.norm(nose_tip - right_eye)
        if (left_dist + right_dist) > 0:
            yaw = np.arctan2(
                right_dist - left_dist,
                (right_dist + left_dist) / 2
            )
        else:
            yaw = 0.0

        # Clamp to OpenFace-like range (-1.5 to 1.5 radians)
        pitch = np.clip(-pitch, -1.5, 1.5)
        yaw   = np.clip(yaw,    -1.5, 1.5)

        # Display in degrees for readability
        cv2.putText(frame, f"Pitch: {np.degrees(pitch):.1f}deg", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Yaw:   {np.degrees(yaw):.1f}deg",   (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return pitch, yaw, None, frame

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

    print("Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pitch, yaw, _, annotated = extractor.get_head_pose(frame)

        if pitch is not None:
            extractor.update_buffer(pitch, yaw)
            cv2.putText(annotated, f"Buffer: {len(extractor.pose_buffer)}/{WINDOW_SIZE}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            # Print raw values so we can verify the range
            print(f"Pitch: {np.degrees(pitch):+.1f}deg  Yaw: {np.degrees(yaw):+.1f}deg", end='\r')

        cv2.imshow("Head Pose Extraction", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.release()

    if len(extractor.pose_buffer) >= 10:
        df = extractor.get_sequence_df()
        df.to_csv("sample_pose_output.csv", index=False)
        print(f"\nSaved {len(df)} rows to sample_pose_output.csv")
        print(df.describe())