import os
import cv2
import joblib
import numpy as np
import mediapipe as mp
import base64
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = 'posture_model.joblib'
SENSITIVITY_THRESHOLD = 0.3  # For Side View (ML)
FRONT_SENSITIVITY = 0.15     # For Front View (Math) - Adjust if needed

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
    """Calculates angle for Side View logic"""
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

    # --- 1. DETECT ORIENTATION (FRONT vs SIDE) ---
    # We check the horizontal distance between shoulders.
    # If shoulders are far apart (wide), it's FRONT view.
    # If shoulders are close together (overlapping), it's SIDE view.
    
    left_shoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
    right_shoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
    
    shoulder_width = abs(left_shoulder_x - right_shoulder_x)
    
    # Threshold: If shoulder width is > 0.15 (normalized coords), likely facing front
    is_front_facing = shoulder_width > 0.15

    # --- 2. BRANCH LOGIC ---
    
    if is_front_facing:
        # === FRONT VIEW LOGIC (GEOMETRIC) ===
        # We can't use the ML model here. We use math rules.
        
        # Rule 1: Shoulder Alignment (Are you leaning?)
        left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        shoulder_tilt = abs(left_shoulder_y - right_shoulder_y)

        # Rule 2: Vertical Head Position (Are you slouching down?)
        # Compare Nose Y to average Shoulder Y
        nose_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
        avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
        
        # Calculate vertical distance (Head to Shoulders)
        # Normal upright posture has a larger gap. Slouching makes this gap smaller.
        head_neck_dist = avg_shoulder_y - nose_y

        print(f"FRONT MODE | Tilt: {shoulder_tilt:.3f} | Neck Dist: {head_neck_dist:.3f}")

        # LOGIC: 
        # - If tilt > 0.04: Leaning too much
        # - If head_neck_dist < 0.15: Head is too close to shoulders (Slumping)
        # (Note: These thresholds might need tuning based on your webcam distance)
        
        if shoulder_tilt > 0.05 or head_neck_dist < FRONT_SENSITIVITY:
            return {'status': 'success', 'result': 'Incorrect Posture (Front)', 'color': 'red', 'mode': 'Front'}
        else:
            return {'status': 'success', 'result': 'Correct Posture (Front)', 'color': 'green', 'mode': 'Front'}

    else:
        # === SIDE VIEW LOGIC (ML MODEL) ===
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
            
            features = [neck_angle, back_angle]
            
            # Predict
            probabilities = model.predict_proba([features])[0]
            prob_incorrect = probabilities[1]
            
            print(f"SIDE MODE | Bad Prob: {prob_incorrect:.2f}")

            if prob_incorrect > SENSITIVITY_THRESHOLD:
                return {'status': 'success', 'result': 'Incorrect Posture (Side)', 'color': 'red', 'mode': 'Side'}
            else:
                return {'status': 'success', 'result': 'Correct Posture (Side)', 'color': 'green', 'mode': 'Side'}

        except Exception as e:
            # Fallback if landmarks are missing (e.g. knees hidden)
            return {'status': 'error', 'message': "Please show full body (Side view)"}


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
        
        if result['status'] == 'no_pose':
             return jsonify({'status': 'no_pose', 'message': 'No person detected'})
        elif result['status'] == 'error':
             return jsonify({'status': 'error', 'message': result['message']})
        
        return jsonify(result)

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)