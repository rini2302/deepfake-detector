Deepfake Detection Engine

A web-based tool to analyze video files and determine the likelihood of them being a deepfake. This project uses a sophisticated Python pipeline with multiple feature extractors and a machine learning model to provide a real/fake probability score. Here, the model is also from scratch trained, and the accuracy is approximately 88-89%.

Features

Simple Web Interface: Easy-to-use UI for uploading and analyzing videos.
Asynchronous Processing: Upload a video and get results on the same page without a refresh.
Multi-Feature Analysis: The backend pipeline extracts various biometric and visual artifact features, including:
    * Blink Detection
    * Skin Tone Variation
    * Facial Geometry
    * Head Pose Dynamics
    * Lip Sync Cohesion
ML-Powered Prediction: Uses a pre-trained stacking model to classify videos as "Real" or "Fake" with a confidence score.

Technology Stack

* Backend: Python, Flask
* Frontend: HTML, CSS, JavaScript (Fetch API)
* Machine Learning: Scikit-learn, Pandas, Joblib
* Computer Vision: OpenCV, Dlib, MediaPipe, InsightFace

Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

Prerequisites

* Python 3.9+
* pip (Python package installer)
* Git

Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/deepfake-detector.git](https://github.com/your-username/deepfake-detector.git)
    cd deepfake-detector
    ```

2.  **Create a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Model Files:**
    Place your pre-trained model files (`shape_predictor_68_face_landmarks.dat` and `stacking_model_tuned.pkl`) into the `models/` directory.

Usage

To run the application locally, execute the `app.py` script from the root directory:

```bash
python app.py
```

Open your web browser and navigate to `http://127.0.0.1:5000`. You can now upload a video to start the analysis.

Project Structure

├── app.py                  # Flask web server
├── pipeline.py             # Core deepfake detection logic
├── requirements.txt        # Python package dependencies
├── packages.txt            # System-level dependencies for Vercel
├── vercel.json             # Vercel deployment configuration
├── .gitignore              # Files to be ignored by Git
├── models/                 # Directory for ML models
│   └── ...
└── templates/
    └── index.html          # Frontend HTML and JavaScript
