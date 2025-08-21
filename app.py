# app.py

import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

# Import your main analysis function from your pipeline
from pipeline import analyze_video

# Initialize the Flask application
app = Flask(__name__)

# --- Configuration ---
# Create a directory to temporarily store uploaded videos
# New version for Vercel
# Use the '/tmp' directory for temporary file storage on Vercel
UPLOAD_FOLDER = '/tmp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Optional: Set a maximum file size, e.g., 100 MB
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# --- Routes ---

@app.route('/')
def index():
    """
    Renders the main page of the web application.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the video upload and prediction process.
    """
    # 1. Check if a file was sent in the request
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided in the request.'}), 400
    
    file = request.files['video']
    
    # 2. Check if the user selected a file
    if file.filename == '':
        return jsonify({'error': 'No video file was selected.'}), 400
        
    # 3. If a file is present and has a name
    if file:
        # Sanitize the filename to prevent security issues
        filename = secure_filename(file.filename)
        # Create the full path to save the video temporarily
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            # 4. Save the video file to the server's filesystem
            file.save(video_path)
            print(f"✅ Video saved temporarily to: {video_path}")

            # 5. Call your analysis pipeline with the path to the video
            print("🚀 Starting deepfake analysis pipeline...")
            result = analyze_video(video_path)
            
            # 6. Return the result from your pipeline as a JSON response
            return jsonify(result)
            
        except Exception as e:
            # Handle any errors that occur during the analysis
            print(f"❌ An error occurred during processing: {e}")
            return jsonify({'error': f'An error occurred during analysis: {str(e)}'}), 500
            
        finally:
            # 7. Clean up: Delete the temporary video file after processing
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"🗑️ Temporary file deleted: {video_path}")
                
    # Fallback error response
    return jsonify({'error': 'An unknown server error occurred.'}), 500

# --- Main execution point ---

if __name__ == '__main__':
    # Run the Flask app
    # `debug=True` is helpful for development but should be `False` in production
    app.run(debug=True)