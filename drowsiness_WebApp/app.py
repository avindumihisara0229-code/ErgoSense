"""
Drowsiness Detection - Flask Backend
=====================================
ErgoSense: AI Wellness Monitor
Component: Drowsiness Detection (Hybrid CNN + Decision Tree + OR Logic)

Models Required (place in /models folder):
  - eye_cnn_model.keras  (CNN for eye open/closed)
  - yawn_model.pkl       (Decision Tree for yawn detection)
  - face_landmarker.task (auto-downloaded on first run)

"""

import os
import cv2
import base64
import pickle
import time
import urllib.request
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# ---- TensorFlow (suppress info logs) --------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras

# ---- MediaPipe (new Tasks API for v0.10.30+) ------------------
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ------------------------------------------------
# App Setup
# ------------------------------------------------
app = Flask(__name__)
CORS(app)

# ---- Paths ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

EYE_MODEL_PATH = os.path.join(MODEL_DIR, "eye_cnn_model.keras")
YAWN_MODEL_PATH = os.path.join(MODEL_DIR, "yawn_model.pkl")
FACE_LANDMARKER_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")

FACE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

# ------------------------------------------------
# Load Models
# ------------------------------------------------
eye_model = None
yawn_model = None
face_landmarker = None


def download_face_landmarker():
    """Download the MediaPipe face landmarker model if not present."""
    if os.path.exists(FACE_LANDMARKER_PATH):
        return True
    try:
        print("[..] Downloading face_landmarker.task (first time only)...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH)
        print("[OK] face_landmarker.task downloaded successfully")
        return True
    except Exception as e:
        print(f"[!!] Failed to download face_landmarker.task: {e}")
        return False


def load_models():
    """Load both models at startup."""
    global eye_model, yawn_model, face_landmarker

    if os.path.exists(EYE_MODEL_PATH):
        eye_model = keras.models.load_model(EYE_MODEL_PATH)
        print(f"[OK] Eye CNN model loaded from {EYE_MODEL_PATH}")
    else:
        print(f"[!!] Eye CNN model NOT found at {EYE_MODEL_PATH}")

    if os.path.exists(YAWN_MODEL_PATH):
        with open(YAWN_MODEL_PATH, "rb") as f:
            yawn_model = pickle.load(f)
        print(f"[OK] Yawn Decision Tree loaded from {YAWN_MODEL_PATH}")
    else:
        print(f"[!!] Yawn model NOT found at {YAWN_MODEL_PATH}")

    if download_face_landmarker():
        base_options = mp_python.BaseOptions(
            model_asset_path=FACE_LANDMARKER_PATH
        )
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
        )
        face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        print("[OK] MediaPipe Face Landmarker initialized (Tasks API)")
    else:
        print("[!!] MediaPipe Face Landmarker NOT available")


# ------------------------------------------------
# MAR (Mouth Aspect Ratio) Helpers
# ------------------------------------------------
MOUTH_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405,
    321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
]


def _dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_mar(mouth_pts):
    """Mouth Aspect Ratio from 20 landmark points."""
    v1 = _dist(mouth_pts[2], mouth_pts[10])
    v2 = _dist(mouth_pts[4], mouth_pts[8])
    v3 = _dist(mouth_pts[6], mouth_pts[14])
    h = _dist(mouth_pts[0], mouth_pts[12])
    if h == 0:
        return 0.0
    return (v1 + v2 + v3) / (3.0 * h)


def extract_mar(image_bgr):
    """Return MAR value from a BGR face image, or None."""
    if face_landmarker is None:
        return None
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = face_landmarker.detect(mp_image)
    if not result.face_landmarks or len(result.face_landmarks) == 0:
        return None
    landmarks = result.face_landmarks[0]
    h, w = image_bgr.shape[:2]
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in MOUTH_LANDMARKS]
    return calculate_mar(pts)


# ------------------------------------------------
# Temporal Tracker
# ------------------------------------------------
class TemporalTracker:
    FPS_ASSUMED = 10
    CLOSURE_SEC = 3

    def __init__(self):
        self.eye_closed_frames = 0
        self.threshold = self.CLOSURE_SEC * self.FPS_ASSUMED

    def update(self, eye_closed: bool):
        if eye_closed:
            self.eye_closed_frames += 1
        else:
            self.eye_closed_frames = 0

    @property
    def closed_seconds(self):
        return round(self.eye_closed_frames / self.FPS_ASSUMED, 1)

    @property
    def prolonged_closure(self):
        return self.eye_closed_frames >= self.threshold

    def reset(self):
        self.eye_closed_frames = 0


tracker = TemporalTracker()


# ------------------------------------------------
# Eye Cropping — extract INDIVIDUAL eye regions (matches training data)
# ------------------------------------------------
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


def crop_single_eye(image_bgr, landmarks, eye_indices):
    """Crop a SINGLE eye from the face — matches how training data was cropped."""
    h, w = image_bgr.shape[:2]
    xs = [int(landmarks[i].x * w) for i in eye_indices]
    ys = [int(landmarks[i].y * h) for i in eye_indices]
    eye_w = max(xs) - min(xs)
    eye_h = max(ys) - min(ys)
    pad_x = int(eye_w * 0.5)
    pad_y = int(eye_h * 1.0)
    x1 = max(0, min(xs) - pad_x)
    y1 = max(0, min(ys) - pad_y)
    x2 = min(w, max(xs) + pad_x)
    y2 = min(h, max(ys) + pad_y)
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def crop_eyes(image_bgr, landmarks):
    """Crop both eyes individually. Returns list of eye crops."""
    crops = []
    for eye_idx in [LEFT_EYE, RIGHT_EYE]:
        crop = crop_single_eye(image_bgr, landmarks, eye_idx)
        if crop is not None:
            crops.append(crop)
    return crops


def detect_face_landmarks(image_bgr):
    """Detect face landmarks once, reuse for both eye and yawn."""
    if face_landmarker is None:
        return None
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = face_landmarker.detect(mp_image)
    if not result.face_landmarks or len(result.face_landmarks) == 0:
        return None
    return result.face_landmarks[0]


# ------------------------------------------------
# Core Detection Logic
# ------------------------------------------------
def predict_eye(image_bgr, landmarks=None):
    """Return ('Open'|'Closed', confidence). Crops EACH eye individually."""
    if eye_model is None:
        return "unknown", 0.0

    if landmarks is not None:
        eye_crops = crop_eyes(image_bgr, landmarks)
        if eye_crops:
            probs = []
            for crop in eye_crops:
                img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (64, 64)).astype("float32") / 255.0
                img = np.expand_dims(img, axis=0)
                prob = float(eye_model.predict(img, verbose=0)[0][0])
                probs.append(prob)

            avg_prob = max(probs)
            label = "Closed" if avg_prob > 0.5 else "Open"
            confidence = avg_prob if label == "Closed" else 1 - avg_prob
            return label, round(confidence * 100, 1)

    # Fallback: use full image (for uploaded cropped eye images)
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64)).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    prob = float(eye_model.predict(img, verbose=0)[0][0])
    label = "Closed" if prob > 0.5 else "Open"
    confidence = prob if label == "Closed" else 1 - prob
    return label, round(confidence * 100, 1)


def predict_yawn(image_bgr, landmarks=None):
    """Return ('Yawn'|'No-Yawn', MAR value)."""
    if yawn_model is None:
        return "unknown", 0.0

    if landmarks is not None:
        h, w = image_bgr.shape[:2]
        pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in MOUTH_LANDMARKS]
        mar = calculate_mar(pts)
    else:
        mar = extract_mar(image_bgr)

    if mar is None:
        return "no_face", 0.0

    if mar < 0.65:
        return "No-Yawn", round(mar, 4)

    pred = yawn_model.predict([[mar]])[0]
    label = "Yawn" if pred == 1 else "No-Yawn"
    return label, round(mar, 4)


def run_detection(image_bgr, use_temporal=False):
    """Full hybrid detection pipeline."""
    landmarks = detect_face_landmarks(image_bgr)

    eye_label, eye_conf = predict_eye(image_bgr, landmarks)
    yawn_label, mar_val = predict_yawn(image_bgr, landmarks)

    if use_temporal:
        tracker.update(eye_label == "Closed")
        eye_prolonged = tracker.prolonged_closure
        closed_sec = tracker.closed_seconds
    else:
        eye_prolonged = eye_label == "Closed"
        closed_sec = 0

    # --- OR Logic -----------------------------------------
    reasons = []
    if use_temporal and eye_prolonged:
        reasons.append(f"Eyes closed for {closed_sec}s (≥3s)")
    elif not use_temporal and eye_label == "Closed":
        reasons.append("Eyes are closed")

    if yawn_label == "Yawn":
        reasons.append(f"Yawning detected (MAR={mar_val})")

    is_drowsy = len(reasons) > 0

    if len(reasons) >= 2:
        alert = "HIGH"
    elif is_drowsy:
        alert = "WARNING"
    else:
        alert = "ALERT"

    return {
        "is_drowsy": is_drowsy,
        "alert_level": alert,
        "reasons": reasons,
        "eye": {
            "state": eye_label,
            "confidence": eye_conf,
            "closed_seconds": closed_sec,
        },
        "yawn": {
            "state": yawn_label,
            "mar": mar_val,
        },
        "timestamp": time.time(),
    }


# ------------------------------------------------
# Utility: decode images
# ------------------------------------------------
def decode_base64_image(b64_string):
    """Decode a base64 (data-URI or raw) string to a BGR numpy array."""
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    img_bytes = base64.b64decode(b64_string)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ------------------------------------------------
# Routes
# ------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify({
        "status": "running",
        "models": {
            "eye_cnn": eye_model is not None,
            "yawn_dt": yawn_model is not None,
            "mediapipe": face_landmarker is not None,
        },
    })


@app.route("/api/detect-image", methods=["POST"])
def detect_image():
    """Upload an image file for one-shot detection (no temporal logic)."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Could not decode image"}), 400

    result = run_detection(image, use_temporal=False)
    return jsonify(result)


@app.route("/api/detect-frame", methods=["POST"])
def detect_frame():
    """Receive a base64 webcam frame for real-time detection (with temporal logic)."""
    data = request.get_json(silent=True)
    if not data or "frame" not in data:
        return jsonify({"error": "No frame data"}), 400

    image = decode_base64_image(data["frame"])
    if image is None:
        return jsonify({"error": "Could not decode frame"}), 400

    result = run_detection(image, use_temporal=True)
    return jsonify(result)


@app.route("/api/reset", methods=["POST"])
def reset_tracker():
    """Reset the temporal eye-closure tracker."""
    tracker.reset()
    return jsonify({"message": "Tracker reset", "eye_closed_frames": 0})


# ------------------------------------------------
# Main
# ------------------------------------------------
if __name__ == "__main__":
    load_models()
    print("\n" + "=" * 55)
    print("  ErgoSense — Drowsiness Detection Component")
    print("  http://127.0.0.1:5001")
    print("=" * 55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5001)