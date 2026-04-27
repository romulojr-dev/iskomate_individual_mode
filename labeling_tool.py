"""
Chunk-level video labeling tool for engagement annotation.
Shows each 280-frame chunk and asks annotator to assign a label.

Run it like this for each subject:
python labeling_tool.py path/to/subject_01_video.mp4 S01 annotator_1
"""
import cv2
import pandas as pd
import os
import json
from datetime import datetime

CHUNK_SIZE = 280  # frames
FPS = 30  # assumed webcam FPS

ENGAGEMENT_LABELS = {
    '0': 'Not Engaged',
    '1': 'Barely Engaged',
    '2': 'Engaged',
    '3': 'Highly Engaged',
    's': 'SKIP this chunk',
}

def label_video(video_path, subject_id, annotator_id, output_dir='annotations'):
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    n_chunks = total_frames // CHUNK_SIZE
    
    print(f"\nVideo: {video_path}")
    print(f"Total frames: {total_frames} | FPS: {actual_fps:.1f} | Chunks: {n_chunks}")
    print(f"Each chunk = {CHUNK_SIZE/actual_fps:.1f} seconds of video\n")
    
    annotations = []
    chunk_idx = 0
    
    while chunk_idx < n_chunks:
        # Seek to start of chunk
        start_frame = chunk_idx * CHUNK_SIZE
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        print(f"\n{'='*50}")
        print(f"Chunk {chunk_idx+1}/{n_chunks} "
              f"(frames {start_frame}-{start_frame+CHUNK_SIZE}, "
              f"~{start_frame/actual_fps:.1f}s - {(start_frame+CHUNK_SIZE)/actual_fps:.1f}s)")
        print("Controls: SPACE=pause/play | R=replay | 0/1/2/3=label | S=skip | Q=quit")
        print(f"Labels: {ENGAGEMENT_LABELS}")
        
        # Play the chunk
        frames = []
        for _ in range(CHUNK_SIZE):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        if not frames:
            break
            
        # Playback loop
        label = None
        while label is None:
            paused = False
            frame_idx = 0
            
            while frame_idx < len(frames):
                frame = frames[frame_idx].copy()
                
                # Overlay info
                cv2.putText(frame, f"Subject: {subject_id} | Chunk: {chunk_idx+1}/{n_chunks}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                cv2.putText(frame, f"Time: {(start_frame+frame_idx)/actual_fps:.1f}s",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                cv2.putText(frame, "Press 0/1/2/3 to label | R to replay | S to skip",
                           (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (200,200,200), 1)
                
                if paused:
                    cv2.putText(frame, "PAUSED", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                
                cv2.imshow(f"Labeling Tool - {subject_id}", frame)
                key = cv2.waitKey(int(1000/actual_fps)) & 0xFF
                
                if key == ord('q'):
                    # Save progress and quit
                    cap.release()
                    cv2.destroyAllWindows()
                    _save_annotations(annotations, subject_id, 
                                     annotator_id, output_dir)
                    print("\nProgress saved. You can resume later.")
                    return annotations
                    
                elif key == ord(' '):
                    paused = not paused
                    
                elif key == ord('r'):
                    # Replay chunk
                    break
                    
                elif key in [ord('0'), ord('1'), ord('2'), ord('3')]:
                    label = chr(key)
                    break
                    
                elif key == ord('s'):
                    label = 's'
                    break
                
                if not paused:
                    frame_idx += 1
            
            if key == ord('r'):
                continue  # replay
            if label is not None:
                break
        
        if label == 's':
            print(f"  → Chunk {chunk_idx+1} SKIPPED")
        else:
            label_name = ENGAGEMENT_LABELS[label]
            print(f"  → Chunk {chunk_idx+1} labeled: {label} ({label_name})")
            annotations.append({
                'subject_id': subject_id,
                'chunk_idx': chunk_idx,
                'start_frame': start_frame,
                'end_frame': start_frame + CHUNK_SIZE,
                'start_time': start_frame / actual_fps,
                'end_time': (start_frame + CHUNK_SIZE) / actual_fps,
                'label': int(label),
                'label_name': label_name,
                'annotator': annotator_id,
                'timestamp': datetime.now().isoformat()
            })
        
        chunk_idx += 1
    
    cap.release()
    cv2.destroyAllWindows()
    _save_annotations(annotations, subject_id, annotator_id, output_dir)
    return annotations

def _save_annotations(annotations, subject_id, annotator_id, output_dir):
    if not annotations:
        return
    df = pd.DataFrame(annotations)
    out_path = f"{output_dir}/{subject_id}_{annotator_id}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nAnnotations saved to {out_path}")
    print(f"Total chunks labeled: {len(df)}")
    print(f"Label distribution:\n{df['label_name'].value_counts()}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python labeling_tool.py <video_path> <subject_id> <annotator_id>")
        print("Example: python labeling_tool.py videos/subject_01.mp4 S01 annotator_1")
        sys.exit(1)
    
    video_path  = sys.argv[1]
    subject_id  = sys.argv[2]
    annotator_id = sys.argv[3]
    
    label_video(video_path, subject_id, annotator_id)