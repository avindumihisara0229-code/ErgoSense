🧘 ErgoSense: AI Wellness Monitor
Privacy-First Real-Time Ergonomic & Stress Monitoring System

📖 About The Project
ErgoSense is a smart desktop application designed to act as a personal wellness coach for desk workers. Using a standard webcam and microphone, it employs four simultaneous AI models to monitor physical and mental well-being in real-time.

Unlike cloud-based solutions, ErgoSense runs 100% locally (Edge AI), ensuring that no video, audio, or biometric data ever leaves the user's device.



Setup Instructions
1. Create a Virtual Environment (Highly Recommended)

Creating a virtual environment ensures that the necessary dependencies and libraries are isolated from the global Python environment, preventing version conflicts and preserving project integrity.

For Windows:
python -m venv venv
venv\Scripts\activate
For Mac/Linux:
python3 -m venv venv
source venv/bin/activate
2. Install Exact Dependencies

Once the virtual environment is activated, create a requirements.txt file in your project directory and paste the following contents:

Flask==3.1.2
opencv-python==4.13.0.92
numpy==1.26.4
mediapipe==0.10.14
joblib==1.5.2
librosa==0.11.0
static_ffmpeg==3.0
tensorflow==2.16.1
keras==3.12.0
scikit-learn==1.8.0

To install the dependencies, run the following command in your terminal:

pip install -r requirements.txt
3. Initialize FFmpeg (For Vocal Stress)

The static_ffmpeg library is used to handle audio chunks for vocal stress detection. Run the following command to download and set up the necessary binaries:

static_ffmpeg -version

This command ensures that static_ffmpeg is properly installed and ready to use for vocal stress analysis.

4. Run the Application

After setting up the environment and dependencies, you can start the Flask web server to run the ErgoSense application.

Run the following command:

python app.py

Once the server is running, navigate to the following URL in your web browser to access the application:

http://127.0.0.1:5000
Required Project Structure

Ensure that your project files are structured as follows for the application to run properly:

/ergosense
│-- app.py
│-- requirements.txt
│-- posture_model.joblib
│-- vocal_stress.joblib
│-- stress_detection_model.joblib
│-- scaler.pkl
│-- label_encoder.pkl
├── /models
│   ├── yawn_model.pkl
│   └── eye_cnn_model.keras
├── /templates
│   ├── index.html
│   └── analytics.html

Note that the following files will be automatically generated when you run the app:

user_baseline.json

ergosense.db
