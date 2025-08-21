# glcm_features.py

import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_glcm_std_features(video_path, distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Extracts GLCM contrast and correlation standard deviations from frames in a folder.

    Args:
        video_path (str): Path to folder containing grayscale image frames of a video.
        distances (list): Pixel pair distance offsets.
        angles (list): Angles (in radians) for GLCM computation.

    Returns:
        dict: A dictionary containing contrast_std, correlation_std
    """
    contrast_vals = []
    correlation_vals = []

    if not os.path.isdir(video_path):
        raise FileNotFoundError(f"❌ Folder not found: {video_path}")

    for frame_file in sorted(os.listdir(video_path)):
        frame_path = os.path.join(video_path, frame_file)
        image = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        glcm = graycomatrix(image, distances=distances, angles=angles, levels=256,
                            symmetric=True, normed=True)

        contrast = np.mean(graycoprops(glcm, 'contrast'))
        correlation = np.mean(graycoprops(glcm, 'correlation'))

        contrast_vals.append(contrast)
        correlation_vals.append(correlation)

    # Calculate standard deviations
    if len(contrast_vals) > 1:
        std_contrast = np.std(contrast_vals)
        std_correlation = np.std(correlation_vals)
    else:
        std_contrast = 0.0
        std_correlation = 0.0

    return {
        "contrast_std": std_contrast,
        "correlation_std": std_correlation
    }
