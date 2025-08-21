import os
import cv2
import dlib
import numpy as np
from mtcnn import MTCNN
from scipy.spatial import distance as dist
from pathlib import Path

class BlinkDetector:
    def __init__(self, predictor_path, ear_threshold=0.21, consec_frames=4):
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"Shape predictor not found at {predictor_path}")
        self.detector = MTCNN()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames

    @staticmethod
    def eye_aspect_ratio(eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def detect_blinks(self, video_path):
        cap = cv2.VideoCapture(video_path)
        blink_count = 0
        consec_counter = 0
        cooldown_frames = 0
        face_found = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.detector.detect_faces(rgb)

            if faces:
                face_found = True
                face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
                x, y, w, h = face['box']
                rect = dlib.rectangle(x, y, x + w, y + h)
                shape = self.predictor(rgb, rect)

                # This block replaces face_utils.shape_to_np(shape)
                coords = np.zeros((shape.num_parts, 2), dtype="int")
                for i in range(0, shape.num_parts):
                    coords[i] = (shape.part(i).x, shape.part(i).y)

                shape = coords

                leftEye = shape[42:48]
                rightEye = shape[36:42]
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                if cooldown_frames > 0:
                    cooldown_frames -= 1
                    continue

                if ear < self.ear_threshold:
                    consec_counter += 1
                else:
                    if consec_counter >= self.consec_frames:
                        blink_count += 1
                        cooldown_frames = 7
                    consec_counter = 0

        cap.release()
        if not face_found:
            return 'NA'
        return blink_count

    @staticmethod
    def get_label_from_path(video_path):
        parts = Path(video_path).parts
        for part in parts:
            if part.lower() == 'real':
                return 'real'
            elif part.lower() == 'fake':
                return 'fake'
        return 'unknown'
