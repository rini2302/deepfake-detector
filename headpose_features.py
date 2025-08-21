# headpose_features.py

import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# === Hopenet model definition ===
class Hopenet(nn.Module):
    def __init__(self, num_bins=66):
        super(Hopenet, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet50.children())[:-1])
        self.fc_yaw = nn.Linear(resnet50.fc.in_features, num_bins)
        self.fc_pitch = nn.Linear(resnet50.fc.in_features, num_bins)
        self.fc_roll = nn.Linear(resnet50.fc.in_features, num_bins)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        yaw = self.fc_yaw(x)
        pitch = self.fc_pitch(x)
        roll = self.fc_roll(x)
        return yaw, pitch, roll

# === Load model and setup ===
device = torch.device('cpu')
idx_tensor = torch.FloatTensor([idx for idx in range(66)]).to(device)

model = Hopenet().to(device)
model.eval()

# === Image preprocessing ===
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def get_headpose(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(device)
        yaw, pitch, roll = model(img_tensor)

        yaw_pred = torch.sum(torch.softmax(yaw, dim=1) * idx_tensor, dim=1) * 3 - 99
        pitch_pred = torch.sum(torch.softmax(pitch, dim=1) * idx_tensor, dim=1) * 3 - 99
        roll_pred = torch.sum(torch.softmax(roll, dim=1) * idx_tensor, dim=1) * 3 - 99

        return yaw_pred.item(), pitch_pred.item(), roll_pred.item()
    except Exception as e:
        print(f"[!] Headpose error on {image_path}: {e}")
        return None

def extract_headpose_dynamics(video_path, frame_skip=1):
    """
    Extracts head pose dynamic features from a video.

    Args:
        video_path (str): Path to a .mp4 video file
        frame_skip (int): Skip every N frames (default=1 for all frames)

    Returns:
        dict: STD of velocity and acceleration for yaw, pitch, roll
    """
    yaws, pitches, rolls = [], [], []
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    temp_path = "temp_headpose.jpg"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip == 0:
            cv2.imwrite(temp_path, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pose = get_headpose(temp_path)
            if pose:
                yaw, pitch, roll = pose
                yaws.append(yaw)
                pitches.append(pitch)
                rolls.append(roll)
        frame_idx += 1
    cap.release()

    if os.path.exists(temp_path):
        os.remove(temp_path)

    if len(yaws) < 3:
        return {
            "yaw_vel_std": 0.0,
            "pitch_vel_std": 0.0,
            "roll_vel_std": 0.0,
            "yaw_acc_std": 0.0,
            "pitch_acc_std": 0.0,
            "roll_acc_std": 0.0,
            "valid": False
        }

    yaw_vel = np.diff(yaws)
    pitch_vel = np.diff(pitches)
    roll_vel = np.diff(rolls)

    yaw_acc = np.diff(yaw_vel)
    pitch_acc = np.diff(pitch_vel)
    roll_acc = np.diff(roll_vel)

    return {
        "yaw_vel_std": round(np.std(yaw_vel), 3),
        "pitch_vel_std": round(np.std(pitch_vel), 3),
        "roll_vel_std": round(np.std(roll_vel), 3),
        "yaw_acc_std": round(np.std(yaw_acc), 3),
        "pitch_acc_std": round(np.std(pitch_acc), 3),
        "roll_acc_std": round(np.std(roll_acc), 3),
        "valid": True
    }
