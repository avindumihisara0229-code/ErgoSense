import os
import cv2
import joblib
import numpy as np
import mediapipe as mp
import base64
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --- CONFIGURATION ---
POSTURE_MODEL_PATH = 'posture_model.joblib'
STRESS_MODEL_PATH = 'stress_model_real.joblib'
SENSITIVITY_THRESHOLD = 0.3  # Side View Posture
FRONT_SENSITIVITY = 0.15     # Front View Posture

# --- LOAD MODELS ---
try:
    posture_model = joblib.load(POSTURE_MODEL_PATH)
    stress_model = joblib.load(STRESS_MODEL_PATH)
    print("✅ Posture and Stress models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    posture_model = None
    stress_model = None

# --- MEDIAPIPE INITIALIZATION ---
mp_pose = mp.solutions.pose
pose_estimator = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

mp_face_mesh = mp.solutions.face_mesh
# static_image_mode=False is better for real-time video streams
face_mesh_estimator = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    min_detection_confidence=0.5
)

# --- HELPER FUNCTIONS ---

def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def analyze_frame(image):
    if image is None: 
        return {'status': 'error', 'message': 'Invalid image'}

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # --- A. POSTURE DETECTION ---
    pose_results = pose_estimator.process(image_rgb)
    posture_data = {"result": "No person detected", "color": "gray", "mode": "N/A"}
    
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        
        # Determine View (Front vs Side)
        ls_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
        rs_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
        is_front_facing = abs(ls_x - rs_x) > 0.15

        if is_front_facing:
            # Front Logic (Mathematical)
            ls_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            rs_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            nose_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
            
            tilt = abs(ls_y - rs_y)
            dist = ((ls_y + rs_y) / 2) - nose_y

            if tilt > 0.05 or dist < FRONT_SENSITIVITY:
                posture_data = {"result": "Incorrect Posture", "color": "red", "mode": "Front"}
            else:
                posture_data = {"result": "Correct Posture", "color": "green", "mode": "Front"}
        else:
            # Side Logic (ML Model)
            try:
                l_sh = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                l_er = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
                l_hp = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                l_kn = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                n_ang = calculate_angle(l_er, l_sh, l_hp)
                b_ang = calculate_angle(l_sh, l_hp, l_kn)
                
                prob_bad = posture_model.predict_proba([[n_ang, b_ang]])[0][1]
                posture_data = {
                    "result": "Incorrect Posture" if prob_bad > SENSITIVITY_THRESHOLD else "Correct Posture",
                    "color": "red" if prob_bad > SENSITIVITY_THRESHOLD else "green",
                    "mode": "Side"
                }
            except:
                posture_data = {"result": "Side view obscured", "color": "orange", "mode": "Side"}

    # --- B. STRESS DETECTION ---
    stress_results = face_mesh_estimator.process(image_rgb)
    stress_data = {"result": "Face not detected", "color": "gray"}

    if stress_results.multi_face_landmarks and stress_model:
        face_landmarks = stress_results.multi_face_landmarks[0].landmark
        
        # Flatten landmarks: exactly 1404 features (468 points * 3 coords)
        features = []
        for l in face_landmarks:
            features.extend([l.x, l.y, l.z])
        
        # Reshape for scikit-learn (1 sample, 1404 features)
        features_arr = np.array(features).reshape(1, -1)
        
        try:
            prediction = stress_model.predict(features_arr)[0]
            stress_data = {
                "result": "Stressed" if prediction == 1 else "Relaxed",
                "color": "red" if prediction == 1 else "green"
            }
        except Exception as e:
            print(f"Stress Prediction Error: {e}")
            stress_data = {"result": "ML Error", "color": "orange"}

    return {'status': 'success', 'posture': posture_data, 'stress': stress_data}

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    try:
        data = request.json['image']
        _, encoded = data.split(",", 1)
        binary_data = base64.b64decode(encoded)
        image_arr = np.frombuffer(binary_data, dtype=np.uint8)
        img = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)

        result = analyze_frame(img)
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)