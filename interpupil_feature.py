# interpupil_feature.py

import os
import cv2
import numpy as np
import mediapipe as mp

# MediaPipe constants for pupil centers
LEFT_EYE_CENTER = 468
RIGHT_EYE_CENTER = 473

# Initialize once at module level
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

def get_interpupil_distance(image):
    """
    Extracts interpupil distance from a face using MediaPipe face mesh.
    
    Args:
        image (np.ndarray): BGR image.

    Returns:
        float or None: Distance between pupil centers, or None if not found.
    """
    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    try:
        left = landmarks[LEFT_EYE_CENTER]
        right = landmarks[RIGHT_EYE_CENTER]
    except IndexError:
        return None

    lx, ly = int(left.x * w), int(left.y * h)
    rx, ry = int(right.x * w), int(right.y * h)

    return np.linalg.norm(np.array([lx, ly]) - np.array([rx, ry]))

def extract_interpupil_std(video_path):
    """
    Computes the standard deviation of interpupil distances across all frames in a video folder.

    Args:
        video_path (str): Path to folder with video frames.

    Returns:
        float: Standard deviation of interpupil distance.
    """
    if not os.path.isdir(video_path):
        raise FileNotFoundError(f"❌ Folder not found: {video_path}")

    pupil_distances = []

    for frame_file in sorted(os.listdir(video_path)):
        frame_path = os.path.join(video_path, frame_file)
        image = cv2.imread(frame_path)
        if image is None:
            continue

        dist = get_interpupil_distance(image)
        if dist is not None:
            pupil_distances.append(dist)

    std_dev = np.std(pupil_distances) if len(pupil_distances) > 1 else 0.0
    return std_dev
