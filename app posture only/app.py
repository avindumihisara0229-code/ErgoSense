import os
import cv2
import joblib
import numpy as np
import mediapipe as mp
import base64
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = 'posture_model.joblib'
SENSITIVITY_THRESHOLD = 0.3  
FRONT_SENSITIVITY = 0.15     

# --- DATABASE SETUP ---
# Connect to SQLite (Creates ergosense.db file automatically)
conn = sqlite3.connect('ergosense.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS posture_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        status TEXT,
        mode TEXT
    )
''')
conn.commit()

# --- CALIBRATION DATA ---
# Stores the user's perfect baseline posture
user_baseline = {
    'front_neck_dist': None,
    'side_neck_angle': None,
    'side_back_angle': None,
    'is_calibrated': False
}

try:
    model = joblib.load(MODEL_PATH)
    print("✅ ERGOSENSE Model loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: '{MODEL_PATH}' not found.")
    model = None

mp_pose = mp.solutions.pose
pose_estimator = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def extract_features(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_estimator.process(image_rgb)
    if not results.pose_landmarks: return None
    return results.pose_landmarks.landmark

def analyze_posture(image):
    landmarks = extract_features(image)
    if not landmarks: return {'status': 'no_pose'}

    # Determine orientation
    left_shoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
    right_shoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
    is_front_facing = abs(left_shoulder_x - right_shoulder_x) > 0.15

    status_result = "Correct"
    posture_mode = "Front" if is_front_facing else "Side"

    if is_front_facing:
        # FRONT VIEW LOGIC
        left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        nose_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
        avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
        head_neck_dist = avg_shoulder_y - nose_y

        if user_baseline['is_calibrated'] and user_baseline['front_neck_dist']:
            # Use personalized calibration (If head drops 20% below baseline = bad)
            if head_neck_dist < (user_baseline['front_neck_dist'] * 0.8):
                status_result = "Incorrect"
        else:
            # Use default math
            if head_neck_dist < FRONT_SENSITIVITY: status_result = "Incorrect"

    else:
        # SIDE VIEW LOGIC
        try:
            ls = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            le = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            lh = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            lk = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            neck_angle = calculate_angle(le, ls, lh)
            back_angle = calculate_angle(ls, lh, lk)

            if user_baseline['is_calibrated'] and user_baseline['side_neck_angle']:
                # Use personalized calibration (Deviation > 15 degrees = bad)
                if abs(neck_angle - user_baseline['side_neck_angle']) > 15 or abs(back_angle - user_baseline['side_back_angle']) > 15:
                    status_result = "Incorrect"
            else:
                # Use ML Model
                prob_incorrect = model.predict_proba([[neck_angle, back_angle]])[0][1]
                if prob_incorrect > SENSITIVITY_THRESHOLD: status_result = "Incorrect"
        except:
             return {'status': 'error', 'message': "Full body not visible"}

    # Save to SQLite Database
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO posture_logs (timestamp, status, mode) VALUES (?, ?, ?)", (timestamp, status_result, posture_mode))
    conn.commit()

    return {
        'status': 'success', 
        'result': f"{status_result} Posture ({posture_mode})", 
        'is_bad': status_result == "Incorrect",
        'calibrated': user_baseline['is_calibrated']
    }

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analytics')
def analytics_page():
    return render_template('analytics.html')

@app.route('/help')
def help_page():
    return render_template('help.html')

@app.route('/calibrate', methods=['POST'])
def calibrate():
    try:
        data = request.json['image']
        header, encoded = data.split(",", 1)
        binary_data = base64.b64decode(encoded)
        img = cv2.imdecode(np.frombuffer(binary_data, dtype=np.uint8), cv2.IMREAD_COLOR)

        landmarks = extract_features(img)
        if not landmarks: return jsonify({'status': 'error', 'message': 'No person detected for calibration'})

        # Calculate Front & Side metrics and save them
        ls_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        rs_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        nose_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
        user_baseline['front_neck_dist'] = ((ls_y + rs_y) / 2) - nose_y

        ls = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, ls_y]
        le = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
        lh = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        lk = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        user_baseline['side_neck_angle'] = calculate_angle(le, ls, lh)
        user_baseline['side_back_angle'] = calculate_angle(ls, lh, lk)

        user_baseline['is_calibrated'] = True
        return jsonify({'status': 'success', 'message': 'Baseline Calibrated Successfully!'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    try:
        data = request.json['image']
        header, encoded = data.split(",", 1)
        img = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), dtype=np.uint8), cv2.IMREAD_COLOR)
        return jsonify(analyze_posture(img))
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/get_history')
def get_history():
    # Fetch the last 100 entries from the database
    cursor.execute("SELECT timestamp, status, mode FROM posture_logs ORDER BY id DESC LIMIT 100")
    rows = cursor.fetchall()
    history = [{'time': r[0], 'status': r[1], 'mode': r[2]} for r in rows]
    return jsonify(history)

if __name__ == '__main__':
    app.run(debug=True)