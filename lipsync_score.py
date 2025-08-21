import os
import cv2
import numpy as np
from mtcnn import MTCNN
import dlib

class LipSyncScoreExtractor:
    def __init__(self, predictor_path):
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"Shape predictor file not found at {predictor_path}")
        self.detector = MTCNN()
        self.predictor = dlib.shape_predictor(predictor_path)

    def extract_face_landmarks(self, frame):
        try:
            result = self.detector.detect_faces(frame)
            if not result:
                return None
            x, y, w, h = result[0]['box']
            face = dlib.rectangle(left=x, top=y, right=x + w, bottom=y + h)
            shape = self.predictor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), face)
            coords = np.array([[p.x, p.y] for p in shape.parts()])
            return coords
        except:
            return None

    def compute_lipsync_score(self, landmarks):
        if landmarks is None or len(landmarks) < 68:
            return -1
        top_lip = landmarks[62]
        bottom_lip = landmarks[66]
        return float(np.linalg.norm(top_lip - bottom_lip) / 10.0)

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "NA"

        scores = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            landmarks = self.extract_face_landmarks(frame)
            score = self.compute_lipsync_score(landmarks)
            if score != -1:
                scores.append(score)

        cap.release()
        if not scores:
            return "NA"
        return round(np.mean(scores), 3)
