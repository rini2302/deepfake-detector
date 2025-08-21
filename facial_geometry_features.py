# facial_geometry_features.py

import os
import cv2
import numpy as np
import mediapipe as mp

# === Initialize MediaPipe Face Mesh once ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# === Facial Landmark Indices ===
LEFT_EYE = 33
RIGHT_EYE = 263
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
LEFT_NOSTRIL = 98
RIGHT_NOSTRIL = 327
NOSE_BRIDGE = 6
NOSE_TIP = 2
LEFT_LIP_CORNER = 61
RIGHT_LIP_CORNER = 291
UPPER_LIP = 13
LOWER_LIP = 14

def get_cheekbone_height(image, h):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0].landmark
    return (abs(lm[LEFT_EYE].y - lm[LEFT_CHEEK].y) + abs(lm[RIGHT_EYE].y - lm[RIGHT_CHEEK].y)) * h / 2

def get_nose_dimensions(image, h, w):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None, None
    lm = results.multi_face_landmarks[0].landmark
    width = abs(lm[RIGHT_NOSTRIL].x - lm[LEFT_NOSTRIL].x) * w
    height = abs(lm[NOSE_TIP].y - lm[NOSE_BRIDGE].y) * h
    return width, height

def get_lip_dimensions(image, h, w):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None, None
    lm = results.multi_face_landmarks[0].landmark
    width = abs(lm[RIGHT_LIP_CORNER].x - lm[LEFT_LIP_CORNER].x) * w
    height = abs(lm[LOWER_LIP].y - lm[UPPER_LIP].y) * h
    return width, height

def extract_facial_geometry_std(video_path):
    """
    Extracts std deviation of facial geometry features from frames in a single video.

    Args:
        video_path (str): Path to folder containing frames.

    Returns:
        dict: Dictionary containing STD values for facial geometry features.
    """
    cheekbone_vals = []
    nose_width_vals, nose_height_vals = [], []
    lip_width_vals, lip_height_vals = [], []

    if not os.path.isdir(video_path):
        raise FileNotFoundError(f"❌ Folder not found: {video_path}")

    for frame_file in sorted(os.listdir(video_path)):
        frame_path = os.path.join(video_path, frame_file)
        image = cv2.imread(frame_path)
        if image is None:
            continue

        h, w, _ = image.shape

        cheekbone = get_cheekbone_height(image, h)
        if cheekbone is not None:
            cheekbone_vals.append(cheekbone)

        nose_w, nose_h = get_nose_dimensions(image, h, w)
        if nose_w is not None:
            nose_width_vals.append(nose_w)
        if nose_h is not None:
            nose_height_vals.append(nose_h)

        lip_w, lip_h = get_lip_dimensions(image, h, w)
        if lip_w is not None:
            lip_width_vals.append(lip_w)
        if lip_h is not None:
            lip_height_vals.append(lip_h)

    return {
        "cheekbone_std": np.std(cheekbone_vals) if len(cheekbone_vals) > 1 else 0.0,
        "nose_width_std": np.std(nose_width_vals) if len(nose_width_vals) > 1 else 0.0,
        "nose_height_std": np.std(nose_height_vals) if len(nose_height_vals) > 1 else 0.0,
        "lip_width_std": np.std(lip_width_vals) if len(lip_width_vals) > 1 else 0.0,
        "lip_height_std": np.std(lip_height_vals) if len(lip_height_vals) > 1 else 0.0,
    }
