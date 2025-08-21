import cv2
import os
import numpy as np

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_keyframes(video_path, output_dir, max_frames=9):
    """
    Extracts keyframes from a video using histogram differences and face detection.
    Saves the frames as JPEGs in the output directory.
    
    Parameters:
        video_path (str): Path to input video file.
        output_dir (str): Directory where extracted frames will be saved.
        max_frames (int): Maximum number of frames to extract.

    Returns:
        List[str]: Paths to the saved frame images.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"⚠️ Could not open video: {video_path}")
            return []

        frames = []
        hist_diffs = []
        face_indices = set()
        last_hist = None
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

            small = cv2.resize(frame, (160, 120))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()

            if last_hist is not None:
                diff = cv2.compareHist(last_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                hist_diffs.append((diff, frame_id))
            last_hist = hist

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) > 0:
                face_indices.add(frame_id)

            frame_id += 1

        cap.release()

        if len(frames) == 0:
            print(f"⚠️ No frames found in video: {video_path}")
            return []

        hist_diffs.sort(key=lambda x: x[0], reverse=True)
        top_hist_ids = [idx for _, idx in hist_diffs[:max_frames]]

        combined = list(set(top_hist_ids + list(face_indices)))
        combined.sort()

        while len(combined) < max_frames:
            step = max(1, len(frames) // (max_frames + 1))
            fill_ids = list(range(0, len(frames), step))[:max_frames]
            combined = list(set(combined + fill_ids))
            combined.sort()

        selected_indices = combined[:max_frames]
        os.makedirs(output_dir, exist_ok=True)

        saved_paths = []
        for i, idx in enumerate(selected_indices):
            if idx < len(frames):
                save_path = os.path.join(output_dir, f"frame_{i+1}.jpg")
                cv2.imwrite(save_path, frames[idx])
                saved_paths.append(save_path)

        return saved_paths

    except Exception as e:
        print(f"❌ Error processing {video_path}: {e}")
        return []
