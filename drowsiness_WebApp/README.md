# ErgoSense — Drowsiness Detection

Drowsiness detection component for the ErgoSense AI Wellness Monitor project. Uses a hybrid approach combining a CNN model for eye state detection and a Decision Tree with MediaPipe for yawn detection.

## How it works

The system takes a face image (from webcam or upload) and runs two checks in parallel:

- **Eye detection** — crops each eye using MediaPipe landmarks, feeds them into a CNN trained on open/closed eye images. If eyes stay closed for 3+ seconds, it flags drowsiness.
- **Yawn detection** — extracts mouth landmarks using MediaPipe, calculates the Mouth Aspect Ratio (MAR), and passes it through a Decision Tree classifier.

Final decision uses OR logic: if either eyes are closed too long OR yawning is detected, the system raises a drowsiness alert.

## Setup

1. Train the models using `Drowsiness_Detection.ipynb` in Google Colab
2. Download `eye_cnn_model.keras` and `yawn_model.pkl` from Google Drive
3. Place both files in the `models/` folder
4. Install dependencies and run:

```
pip install -r requirements.txt
python app.py
```

5. Open `http://127.0.0.1:5001` in your browser

The app will automatically download the MediaPipe face landmarker model on first run.

## Project structure

```
drowsiness_detection/
├── app.py                  # Flask backend with detection API
├── requirements.txt        # Python dependencies
├── README.md
├── models/
│   ├── eye_cnn_model.keras # CNN for eye open/closed
│   └── yawn_model.pkl      # Decision Tree for yawn
└── templates/
    └── index.html          # Frontend UI
```

## API endpoints

- `GET /` — frontend UI
- `GET /api/status` — check if models are loaded
- `POST /api/detect-image` — upload an image for detection
- `POST /api/detect-frame` — send a base64 webcam frame for real-time detection
- `POST /api/reset` — reset the eye closure timer

All API routes return JSON responses that can be consumed by any frontend.

## For team leader integration

The leader can either call the API endpoints from the main ErgoSense backend, or import the detection function directly:

```python
from app import run_detection

result = run_detection(image_bgr, use_temporal=False)
```

This returns a dictionary with `is_drowsy`, `alert_level`, eye state, and yawn state.

## Built with

- TensorFlow/Keras (CNN model)
- scikit-learn (Decision Tree)
- MediaPipe (face landmarks)
- Flask (backend)
- OpenCV (image processing)
- **Method:** Hybrid (CNN for eyes + Decision Tree for yawn + OR logic)
