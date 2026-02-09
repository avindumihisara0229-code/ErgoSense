import os
import cv2
import joblib
import numpy as np
import mediapipe as mp
import base64
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = 'posture_model.joblib'
SENSITIVITY_THRESHOLD = 0.3  # Side View (ML)
FRONT_SENSITIVITY = 0.15     # Front View (Math)

# --- GLOBAL VARIABLES ---
# We use a simple list to store history in memory.
# In a real production app, you would use a database (SQL).
posture_history = [] 

# --- LOAD MODEL ---
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: '{MODEL_PATH}' not found.")
    model = None

mp_pose = mp.solutions.pose
pose_estimator = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# --- HELPER FUNCTIONS ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def analyze_posture(image):
    if image is None: return {'status': 'no_pose'}

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_estimator.process(image_rgb)

    if not results.pose_landmarks:
        return {'status': 'no_pose'}

    landmarks = results.pose_landmarks.landmark

    # 1. Orientation Check
    left_shoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
    right_shoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
    shoulder_width = abs(left_shoulder_x - right_shoulder_x)
    is_front_facing = shoulder_width > 0.15

    status_result = ""
    posture_mode = ""

    # 2. Logic Branch
    if is_front_facing:
        # Front Logic
        left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        shoulder_tilt = abs(left_shoulder_y - right_shoulder_y)

        nose_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
        avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
        head_neck_dist = avg_shoulder_y - nose_y

        posture_mode = "Front"
        if shoulder_tilt > 0.05 or head_neck_dist < FRONT_SENSITIVITY:
            status_result = "Incorrect"
        else:
            status_result = "Correct"

    else:
        # Side Logic
        try:
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_ear      = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x, 
                             landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            left_hip      = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee     = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            neck_angle = calculate_angle(left_ear, left_shoulder, left_hip)
            back_angle = calculate_angle(left_shoulder, left_hip, left_knee)
            
            probabilities = model.predict_proba([[neck_angle, back_angle]])[0]
            prob_incorrect = probabilities[1]
            
            posture_mode = "Side"
            if prob_incorrect > SENSITIVITY_THRESHOLD:
                status_result = "Incorrect"
            else:
                status_result = "Correct"
        except:
             return {'status': 'error', 'message': "Full body not visible"}

    # 3. Log History
    # We only log if it's 'Incorrect' to save space, or you can log everything.
    # Let's log every 10th frame or just return the status to JS to handle logging.
    # Here we will just return the result and let the frontend decide when to alert.
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Store in global list (Limit to last 50 entries to prevent memory overflow)
    posture_history.insert(0, {'time': timestamp, 'status': status_result, 'mode': posture_mode})
    if len(posture_history) > 50:
        posture_history.pop()

    return {
        'status': 'success', 
        'result': f"{status_result} Posture ({posture_mode})", 
        'is_bad': status_result == "Incorrect",
        'color': 'red' if status_result == "Incorrect" else 'green'
    }


# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    try:
        data = request.json['image']
        header, encoded = data.split(",", 1)
        binary_data = base64.b64decode(encoded)
        image_arr = np.frombuffer(binary_data, dtype=np.uint8)
        img = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)

        result = analyze_posture(img)
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/get_history')
def get_history():
    """Returns the latest history log to the webpage"""
    return jsonify(posture_history)

if __name__ == '__main__':
    app.run(debug=True)