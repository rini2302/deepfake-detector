import os
import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Landmark indices
LEFT_CHEEK = [234, 93, 132, 58, 172]
RIGHT_CHEEK = [454, 323, 361, 288, 397]
FOREHEAD = [10, 338, 297, 332, 284, 251, 389, 356]

def extract_skin_tone(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return (0, 0, 0)

    h, w, _ = image.shape
    landmarks = results.multi_face_landmarks[0].landmark

    skin_pixels = []
    all_points = LEFT_CHEEK + RIGHT_CHEEK + FOREHEAD

    for idx in all_points:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        patch = image[max(0, y-2):y+3, max(0, x-2):x+3]
        if patch.size != 0:
            skin_pixels.extend(patch.reshape(-1, 3))

    if not skin_pixels:
        return (0, 0, 0)

    avg_color = np.mean(skin_pixels, axis=0)
    return tuple(avg_color)

def compute_skin_tone_variation_single(video_frames_folder):
    """
    Computes skin tone variation score for a single video based on its cropped frames.

    Parameters:
        video_frames_folder (str): Path to the folder containing video frame images.

    Returns:
        float: Skin tone variation score (0.0 if insufficient data).
    """
    skin_colors = []

    for img_file in sorted(os.listdir(video_frames_folder)):
        if not img_file.lower().endswith('.jpg'):
            continue

        img_path = os.path.join(video_frames_folder, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue

        skin_color = extract_skin_tone(image)
        skin_colors.append(skin_color)

    if len(skin_colors) <= 1:
        return 0.0

    skin_colors_np = np.array(skin_colors)
    std_rgb = np.std(skin_colors_np, axis=0)
    variation_score = np.mean(std_rgb)
    return variation_score
