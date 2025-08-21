# retinaface_crop.py

import os
import cv2
from tqdm import tqdm
from insightface.app import FaceAnalysis

def initialize_retinaface():
    """
    Initializes the RetinaFace model (InsightFace).
    Returns the FaceAnalysis app.
    """
    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])  # Change provider if GPU available
    app.prepare(ctx_id=0)
    return app

def crop_faces_with_retinaface_single(video_dir, app=None):
    """
    Crops faces from frames in a given directory using RetinaFace.
    Deletes frames without detectable faces.

    Args:
        video_dir (str): Path to the folder containing video frames (.jpg).
        app (FaceAnalysis): Pre-initialized RetinaFace model. If None, initializes inside.
    """
    if not os.path.isdir(video_dir):
        print(f"❌ Invalid folder path: {video_dir}")
        return

    if app is None:
        app = initialize_retinaface()

    frames_processed = 0
    faces_saved = 0

    for frame_file in tqdm(os.listdir(video_dir), desc=os.path.basename(video_dir)):
        if not frame_file.lower().endswith('.jpg'):
            continue

        frame_path = os.path.join(video_dir, frame_file)
        image = cv2.imread(frame_path)
        if image is None:
            continue

        faces = app.get(image)

        if len(faces) == 0:
            os.remove(frame_path)  # No face found
            continue

        # Use the first detected face
        bbox = faces[0].bbox.astype(int)
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

        cropped = image[y1:y2, x1:x2]

        if cropped.size == 0:
            os.remove(frame_path)
            continue

        # Overwrite original frame with cropped face
        cv2.imwrite(frame_path, cropped)
        faces_saved += 1
        frames_processed += 1

    print(f"\n📁 Done with: {os.path.basename(video_dir)} | Frames processed: {frames_processed} | Faces saved: {faces_saved}")
    print("✅ All frames processed with RetinaFace. Frames without faces deleted.")
