# pipeline.py

import os
import joblib
import pandas as pd
import tempfile
import shutil

# --- Import all your feature extraction modules ---
from fc_extract import extract_keyframes
from retinaface_crop import crop_faces_with_retinaface_single, initialize_retinaface
from blink_detector import BlinkDetector
from skintone import compute_skin_tone_variation_single
from glcm_features import extract_glcm_std_features
from interpupil_feature import extract_interpupil_std
from facial_geometry_features import extract_facial_geometry_std
from headpose_features import extract_headpose_dynamics
from lipsync_score import LipSyncScoreExtractor

# ==============================================================================
# CONFIGURATION: UPDATE THESE PATHS
# ==============================================================================
# It's best practice to place your model files inside your project folder.
# For example, create a 'models' subfolder.
PREDICTOR_PATH = r"D:\deepf filna\test_train\shape_predictor_68_face_landmarks.dat"
MODEL_PATH = r"D:\DF\codesssssssssss\stacking_model_tuned.pkl"
# ==============================================================================


# ==============================================================================
# MODEL INITIALIZATION (Done only once for efficiency)
# ==============================================================================
print("🚀 Initializing models, please wait...")
try:
    RETINAFACE_APP = initialize_retinaface()
    BLINK_EXTRACTOR = BlinkDetector(PREDICTOR_PATH)
    LIP_SYNC_EXTRACTOR = LipSyncScoreExtractor(PREDICTOR_PATH)
    FINAL_MODEL = joblib.load(MODEL_PATH)
    print("✅ Models initialized successfully.")
except Exception as e:
    print(f"❌ Critical Error: Failed to initialize models. Please check paths. Error: {e}")
    RETINAFACE_APP = None
    BLINK_EXTRACTOR = None
    LIP_SYNC_EXTRACTOR = None
    FINAL_MODEL = None
# ==============================================================================


def analyze_video(video_path):
    """
    The main analysis pipeline function.
    
    Args:
        video_path (str): The path to the video file to be analyzed.

    Returns:
        dict: A dictionary containing 'real_percentage' and 'fake_percentage'.
    """
    if not FINAL_MODEL:
        raise Exception("Analysis model is not loaded. Cannot proceed.")

    # Create a temporary directory to store frames
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_output_folder = os.path.join(temp_dir, "frames")
        os.makedirs(frame_output_folder, exist_ok=True)
        
        print(f"Processing video: {os.path.basename(video_path)}")
        features = {}

        try:
            # 1. Frame Extraction
            print("Step 1/9: Extracting keyframes...")
            frame_paths = extract_keyframes(video_path, frame_output_folder)
            if not frame_paths:
                raise ValueError("Frame extraction failed. The video might be corrupt or in an unsupported format.")
            print(f"✅ Extracted {len(frame_paths)} frames.")

            # 2. Face Cropping
            print("Step 2/9: Cropping faces...")
            crop_faces_with_retinaface_single(frame_output_folder, RETINAFACE_APP)
            print("✅ Faces cropped.")

            # 3. Blink Detection
            print("Step 3/9: Detecting blinks...")
            features["blinks"] = BLINK_EXTRACTOR.detect_blinks(video_path)
            print(f"👁️ Blinks detected: {features['blinks']}")

            # 4. Skin Tone Variation
            print("Step 4/9: Analyzing skin tone...")
            features["skin_tone_var"] = compute_skin_tone_variation_single(frame_output_folder)
            print(f"🎨 Skin tone variation: {features['skin_tone_var']:.4f}")

            # 5. GLCM Features
            print("Step 5/9: Extracting texture features (GLCM)...")
            glcm_results = extract_glcm_std_features(frame_output_folder)
            features["contrast_std"] = glcm_results['contrast_std']
            features["correlation_std"] = glcm_results['correlation_std']
            print(f"🟠 GLCM Contrast STD: {features['contrast_std']:.4f}, Correlation STD: {features['correlation_std']:.4f}")

            # 6. Interpupil Distance
            print("Step 6/9: Calculating interpupil distance...")
            features["interpupil_std"] = extract_interpupil_std(frame_output_folder)
            print(f"👁️ Interpupil Distance STD: {features['interpupil_std']:.4f}")

            # 7. Facial Geometry
            print("Step 7/9: Analyzing facial geometry...")
            geo_results = extract_facial_geometry_std(frame_output_folder)
            features.update({
                "cheekbone_std": geo_results['cheekbone_std'],
                "nose_width_std": geo_results['nose_width_std'],
                "nose_height_std": geo_results['nose_height_std'],
                "lip_width_std": geo_results['lip_width_std'],
                "lip_height_std": geo_results['lip_height_std']
            })
            print("✅ Facial geometry analyzed.")

            # 8. Head Pose Dynamics
            print("Step 8/9: Tracking head pose...")
            pose_results = extract_headpose_dynamics(video_path)
            if pose_results["valid"]:
                features.update({
                    "yaw_vel_std": pose_results["yaw_vel_std"],
                    "pitch_vel_std": pose_results["pitch_vel_std"],
                    "roll_vel_std": pose_results["roll_vel_std"],
                    "yaw_acc_std": pose_results["yaw_acc_std"],
                    "pitch_acc_std": pose_results["pitch_acc_std"],
                    "roll_acc_std": pose_results["roll_acc_std"]
                })
                print("✅ Head pose dynamics extracted.")
            else:
                # Use default values if head pose extraction fails
                features.update({
                    "yaw_vel_std": 0, "pitch_vel_std": 0, "roll_vel_std": 0,
                    "yaw_acc_std": 0, "pitch_acc_std": 0, "roll_acc_std": 0
                })
                print("[!] Warning: Could not extract head pose dynamics; using default values.")

            # 9. Lip Sync Score
            print("Step 9/9: Evaluating lip sync...")
            features["lipsync_score"] = LIP_SYNC_EXTRACTOR.process_video(video_path)
            print(f"👄 Lip sync score: {features['lipsync_score']:.4f}")

            # --- Final Prediction ---
            print("\n🔬 Making final prediction...")
            df = pd.DataFrame([features])
            
            # Ensure the DataFrame columns are in the same order as the model was trained on
            # This is a crucial step to prevent errors. You might need to adjust this list.
            # df = df[MODEL_TRAINING_COLUMNS] 
            
            probabilities = FINAL_MODEL.predict_proba(df)[0]
            real_prob = probabilities[0]
            fake_prob = probabilities[1]
            
            result = {
                'fake_percentage': round(fake_prob * 100, 2),
                'real_percentage': round(real_prob * 100, 2)
            }
            
            print(f"Confidence: {result['real_percentage']:.2f}% Real | {result['fake_percentage']:.2f}% Fake")
            return result

        except Exception as e:
            print(f"❌ An error occurred during pipeline execution: {e}")
            # In case of an error, you might want to return an error state
            # For now, we'll re-raise the exception to be caught by the Flask app
            raise e
        
        # The 'with' statement will automatically clean up the temp_dir