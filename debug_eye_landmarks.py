import cv2
import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\OpenFace\calibration_output\calibration_awake.csv')
row = df.iloc[100]

cap = cv2.VideoCapture(r'C:\OpenFace\calibration_videos\awake.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
cap.release()

# Get left eye landmarks (0-27)
points = {}
for i in range(28):
    x = int(float(row[f' eye_lmk_x_{i}']))
    y = int(float(row[f' eye_lmk_y_{i}']))
    points[i] = (x, y)

# Find bounding box of left eye landmarks
all_x = [p[0] for p in points.values()]
all_y = [p[1] for p in points.values()]
min_x, max_x = min(all_x), max(all_x)
min_y, max_y = min(all_y), max(all_y)

# Crop with generous padding
pad = 40
crop = frame[
    max(0, min_y - pad):max_y + pad,
    max(0, min_x - pad):max_x + pad
]

# Scale up 6x for clarity
scale = 6
crop = cv2.resize(crop, (crop.shape[1]*scale, crop.shape[0]*scale),
                  interpolation=cv2.INTER_LINEAR)

# Draw each landmark on the cropped+scaled image
for i, (x, y) in points.items():
    # Adjust coordinates to crop space then scale
    cx = (x - (min_x - pad)) * scale
    cy = (y - (min_y - pad)) * scale

    # Draw point
    cv2.circle(crop, (cx, cy), 6, (0, 255, 0), -1)

    # Draw index with black background
    label = str(i)
    (tw, th), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

    # Alternate label position up/down to avoid overlap
    if i % 2 == 0:
        lx, ly = cx + 8, cy - 8
    else:
        lx, ly = cx + 8, cy + 18

    cv2.rectangle(crop,
                  (lx - 2, ly - th - 2),
                  (lx + tw + 2, ly + 2),
                  (0, 0, 0), -1)
    cv2.putText(crop, label, (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

cv2.imwrite("left_eye_zoom.png", crop)
print("Saved to left_eye_zoom.png — open this file and zoom in")
cv2.imshow("Left Eye Landmarks", crop)
cv2.waitKey(0)
cv2.destroyAllWindows()