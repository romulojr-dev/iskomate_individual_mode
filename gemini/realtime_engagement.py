import cv2
import sys
import os
import json
import torch
import time
import numpy as np
import pandas as pd
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'TCCT_Net'))

from TCCT_Net.models.feature_fusion import Decision_Fusion
from TCCT_Net.data.data_processing import batch_cwt
from gemini.mediapipe_extractor import HeadPoseExtractor, WINDOW_SIZE

ENGAGEMENT_LABELS = {
    0: ("Not Engaged",    (0, 0, 255)),
    1: ("Barely Engaged", (0, 165, 255)),
    2: ("Engaged",        (0, 255, 255)),
    3: ("Highly Engaged", (0, 255, 0)),
}

AMPLIFICATION_FACTOR = 2.0

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
        
    return predicted_class

def main():
    config = load_config('TCCT_Net/config.json')
    model, device = load_model(config)
    extractor = HeadPoseExtractor()
    cap = cv2.VideoCapture(0)
    
    engagement_class = None
    target_class = None
    frame_count = 0
    PREDICT_EVERY = 10
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
            frame_count += 1

            if frame_count % PREDICT_EVERY == 0 and len(extractor.pose_buffer) >= WINDOW_SIZE:
                try:
                    target_class = predict_engagement(model, device, extractor.pose_buffer, config)
                    
                    if engagement_class is None:
                        engagement_class = target_class
                    elif target_class > engagement_class:
                        engagement_class += 1
                    elif target_class < engagement_class:
                        engagement_class -= 1
                        
                except Exception as e:
                    pass

        if engagement_class is not None:
            label_text, color = ENGAGEMENT_LABELS[engagement_class]
            cv2.putText(annotated, f"Engagement: {label_text}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            buffer_pct = int((len(extractor.pose_buffer) / WINDOW_SIZE) * 100)
            cv2.putText(annotated, f"Warming up... {buffer_pct}%", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Student Engagement Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.release()

if __name__ == "__main__":
    main()