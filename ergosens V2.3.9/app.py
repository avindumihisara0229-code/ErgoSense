import os
import static_ffmpeg
static_ffmpeg.add_paths()
import cv2
import joblib
import numpy as np
import mediapipe as mp
import base64
import sqlite3
import librosa
import tempfile
import json
import pickle
import google.generativeai as genai # NEW: Added for Chatbot
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from tensorflow import keras

app = Flask(__name__)

# --- GEMINI AI SETUP ---
# Securely configure the API key on the backend
genai.configure(api_key="AIzaSyBSFW4SyigNmGwN0GZGGkU01y1ENcfRrak")
# UPDATED to the working model version:
gemini_model = genai.GenerativeModel('gemini-2.0-flash-001')

# --- CONFIGURATION ---
MODEL_PATH = 'posture_model.joblib'
VOCAL_MODEL_PATH = 'vocal_stress.joblib'
# UPDATED: Paths for your newly trained model and preprocessors
STRESS_MODEL_PATH = 'stress_detection_model.joblib' 
SCALER_PATH = 'scaler.pkl'
ENCODER_PATH = 'label_encoder.pkl'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YAWN_MODEL_PATH = os.path.join(BASE_DIR, "models", "yawn_model.pkl")
EYE_MODEL_PATH = os.path.join(BASE_DIR, "models", "eye_cnn_model.keras")

# SENSITIVITY & PERSISTENCE
STRESS_PERSISTENCE_LIMIT = 300 
SENSITIVITY_THRESHOLD = 0.3   
FRONT_SENSITIVITY = 0.15
LOUDNESS_THRESHOLD = 0.006     

# DROWSINESS CONFIG (Using ML + Timers to prevent flicker)
EYE_AR_THRESH = 0.21           # Added back EAR threshold for fallback
MAR_THRESH = 0.8             # Set to 0.60 to prevent talking from triggering yawns
YAWN_SECONDS_LIMIT = 2.5       # MUST open mouth for 2.5 full seconds to trigger
SLOW_BLINK_SECONDS = 1.0       # eye closed 1+ second = slow/drowsy blink
DROWSY_WINDOW = 300            # 5 minutes in seconds
SLOW_BLINK_THRESHOLD = 3       # 3+ slow blinks in 5 min = drowsy

# --- GLOBAL TRACKERS ---
stress_tracker = {'streak': 0}
drowsy_tracker = {
    'yawn_start_time': None,
    'eye_closed_start': None,
    'slow_blinks': [],
}

# --- SAFE DATABASE CONNECTION ---
def get_db_connection():
    conn = sqlite3.connect('ergosense.db', timeout=10)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS posture_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            status TEXT,
            vocal_status TEXT,
            visual_status TEXT,
            drowsy_status TEXT,
            mode TEXT
        )
    ''')
    conn.commit()
    return conn

get_db_connection().close()

# --- INITIALIZE MEDIAPIPE ---
mp_pose = mp.solutions.pose
pose_estimator = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# --- MODEL LOADING ---
eye_model = None
try:
    posture_model = joblib.load(MODEL_PATH)
    v_bundle = joblib.load(VOCAL_MODEL_PATH)
    v_model, v_scaler = v_bundle['model'], v_bundle['scaler']
    
    # Load custom stress model, scaler, and encoder
    stress_model = joblib.load(STRESS_MODEL_PATH) 
    stress_scaler = joblib.load(SCALER_PATH)
    stress_encoder = joblib.load(ENCODER_PATH)
    
    yawn_model = pickle.load(open(YAWN_MODEL_PATH, "rb")) if os.path.exists(YAWN_MODEL_PATH) else None
    if os.path.exists(EYE_MODEL_PATH):
        eye_model = keras.models.load_model(EYE_MODEL_PATH)
        print("✅ Eye CNN Model Loaded!")
    print("✅ All AI Models Loaded Successfully!")
except Exception as e:
    print(f"❌ Model Loading Error: {e}")
    posture_model = v_model = stress_model = stress_scaler = stress_encoder = yawn_model = None

# --- EYE CROPPING (for CNN-based eye detection) ---
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

MOUTH_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405,
    321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
]

def crop_single_eye(image_bgr, lms, eye_indices, h, w):
    """Crop a single eye region from face using landmarks."""
    xs = [int(lms[i].x * w) for i in eye_indices]
    ys = [int(lms[i].y * h) for i in eye_indices]
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

def predict_eye_cnn(image_bgr, lms):
    """Use CNN to detect if eyes are open or closed."""
    if eye_model is None:
        return "Open", 0.0
    
    h, w = image_bgr.shape[:2]
    probs = []
    
    for eye_idx in [LEFT_EYE, RIGHT_EYE]:
        crop = crop_single_eye(image_bgr, lms, eye_idx, h, w)
        if crop is not None:
            img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (64, 64)).astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)
            prob = float(eye_model.predict(img, verbose=0)[0][0])
            probs.append(prob)
    
    if not probs:
        return "Open", 0.0
    
    avg_prob = max(probs)
    label = "Closed" if avg_prob > 0.5 else "Open"
    confidence = avg_prob if label == "Closed" else 1 - avg_prob
    return label, round(confidence * 100, 1)

def get_ear(lms, eye_idxs):
    """Mathematical fallback for eye closure."""
    v1 = np.linalg.norm(np.array([lms[eye_idxs[1]].x, lms[eye_idxs[1]].y]) - np.array([lms[eye_idxs[5]].x, lms[eye_idxs[5]].y]))
    v2 = np.linalg.norm(np.array([lms[eye_idxs[2]].x, lms[eye_idxs[2]].y]) - np.array([lms[eye_idxs[4]].x, lms[eye_idxs[4]].y]))
    h = np.linalg.norm(np.array([lms[eye_idxs[0]].x, lms[eye_idxs[0]].y]) - np.array([lms[eye_idxs[3]].x, lms[eye_idxs[3]].y]))
    return (v1 + v2) / (2.0 * h) if h != 0 else 0

def calculate_mar_dt(lms):
    """Calculate MAR using the 20-point mouth landmarks."""
    def _dist(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    pts = [(lms[i].x, lms[i].y) for i in MOUTH_LANDMARKS]
    v1 = _dist(pts[2], pts[10])
    v2 = _dist(pts[4], pts[8])
    v3 = _dist(pts[6], pts[14])
    h = _dist(pts[0], pts[12])
    if h == 0:
        return 0.0
    return (v1 + v2 + v3) / (3.0 * h)

def predict_yawn_dt(lms):
    """Use Decision Tree to detect yawn."""
    mar = calculate_mar_dt(lms)
    
    if mar < MAR_THRESH:
        return "No-Yawn", round(mar, 4)
    
    if yawn_model is not None:
        pred = yawn_model.predict([[mar]])[0]
        label = "Yawn" if pred == 1 else "No-Yawn"
    else:
        label = "Yawn" if mar >= MAR_THRESH else "No-Yawn"
    
    return label, round(mar, 4)

def extract_facial_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks: return None
    return results.multi_face_landmarks[0].landmark

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# --- BASELINE ---
BASELINE_FILE = 'user_baseline.json'
user_baseline = {
    'front_neck_dist': None, 
    'side_neck_angle': None, 
    'side_back_angle': None,
    'is_calibrated': False, 
    'vocal_threshold': 0.005
}
if os.path.exists(BASELINE_FILE):
    try:
        with open(BASELINE_FILE, 'r') as f: user_baseline.update(json.load(f))
    except: pass

def save_baseline_to_disk():
    with open(BASELINE_FILE, 'w') as f: json.dump(user_baseline, f)

# --- CORE ANALYSIS ENGINE ---
def analyze_multimodal(image):
    global stress_tracker, drowsy_tracker
    lms = extract_facial_landmarks(image)
    
    visual_status = "Optimal"
    drowsy_status = "Alert"
    mar_val = 0.0
    
    if lms:
        # 1. Custom Stress Model Integration
        feats = []
        for l in lms:
            feats.extend([l.x, l.y])
            
        if stress_model and stress_scaler and stress_encoder:
            try:
                scaled_feats = stress_scaler.transform([feats])
                pred_encoded = stress_model.predict(scaled_feats)
                pred_label = stress_encoder.inverse_transform(pred_encoded)[0]
                
                if pred_label == "stress":
                    stress_tracker['streak'] += 1
                    if stress_tracker['streak'] >= STRESS_PERSISTENCE_LIMIT: 
                        visual_status = "Stressed"
                else: 
                    stress_tracker['streak'] = 0
            except Exception as e:
                print(f"Stress Prediction Error: {e}")

        # 2. Drowsiness & Yawn Logic (ML Driven + Math Fallback)
        eye_state, _ = predict_eye_cnn(image, lms)
        ear_val = (get_ear(lms, [33, 160, 158, 133, 153, 144]) + get_ear(lms, [362, 385, 387, 263, 373, 380])) / 2.0
        yawn_state, mar_val = predict_yawn_dt(lms)
        now = datetime.now().timestamp()

        # ML-Based Slow Blink Tracking (CNN OR EAR Fallback)
        is_closed = (eye_state == "Closed") or (ear_val < EYE_AR_THRESH)
        
        if is_closed:
            if drowsy_tracker['eye_closed_start'] is None:
                drowsy_tracker['eye_closed_start'] = now
        else:
            if drowsy_tracker['eye_closed_start'] is not None:
                closed_duration = now - drowsy_tracker['eye_closed_start']
                if closed_duration >= SLOW_BLINK_SECONDS:
                    drowsy_tracker['slow_blinks'].append(now)
                drowsy_tracker['eye_closed_start'] = None
                
        # ML-Based Yawn Temporal Logic
        is_yawning = (yawn_state == "Yawn") and (mar_val >= MAR_THRESH)
        
        if is_yawning:
            if drowsy_tracker['yawn_start_time'] is None:
                drowsy_tracker['yawn_start_time'] = now
            
            yawn_duration = now - drowsy_tracker['yawn_start_time']
            if yawn_duration >= YAWN_SECONDS_LIMIT:
                drowsy_status = "Yawn Detected"
        else:
            drowsy_tracker['yawn_start_time'] = None

        # Check slow blinks history
        drowsy_tracker['slow_blinks'] = [t for t in drowsy_tracker['slow_blinks'] if now - t < DROWSY_WINDOW]
        if len(drowsy_tracker['slow_blinks']) >= SLOW_BLINK_THRESHOLD:
            drowsy_status = "Drowsy"

    # 3. Posture Detection
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_estimator.process(image_rgb)
    p_status, p_mode = "Correct", "Front"
    
    if results.pose_landmarks:
        l = results.pose_landmarks.landmark
        
        is_front = abs(l[11].x - l[12].x) > 0.22
        p_mode = "Front" if is_front else "Side"
        
        if is_front:
            dist = ((l[11].y + l[12].y) / 2) - l[0].y
            limit = user_baseline['front_neck_dist'] * 0.72 if user_baseline['is_calibrated'] else FRONT_SENSITIVITY
            shoulder_tilt = abs(l[11].y - l[12].y)
            
            if dist < limit: p_status = "Incorrect (Slumping)"
            elif shoulder_tilt > 0.08: p_status = "Incorrect (Side Lean)"
        else:
            le, ls, lh, lk = [l[7].x, l[7].y], [l[11].x, l[11].y], [l[23].x, l[23].y], [l[25].x, l[25].y]
            n_ang = calculate_angle(le, ls, lh)
            b_ang = calculate_angle(ls, lh, lk)
            
            if user_baseline['is_calibrated']:
                if abs(n_ang - user_baseline['side_neck_angle']) > 18 or abs(b_ang - user_baseline['side_back_angle']) > 18:
                    p_status = "Incorrect (Side Angle)"
            elif posture_model:
                if posture_model.predict_proba([[n_ang, b_ang]])[0][1] > SENSITIVITY_THRESHOLD: 
                    p_status = "Incorrect"

    # Save and Return
    conn = get_db_connection()
    conn.execute("INSERT INTO posture_logs (timestamp, status, vocal_status, visual_status, drowsy_status, mode) VALUES (?, ?, ?, ?, ?, ?)",
                 (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), p_status, "N/A", visual_status, drowsy_status, p_mode))
    conn.commit(); conn.close()

    return {
        'status': 'success', 'result': f"{p_status} ({p_mode})", 'is_bad': "Incorrect" in p_status,
        'visual': visual_status, 'stress_streak': stress_tracker['streak'],
        'drowsy': drowsy_status, 'drowsy_streak': len(drowsy_tracker['slow_blinks']) if lms else 0,
        'debug_mar': round(mar_val, 3) if lms else 0
    }

# --- REST OF THE ROUTES ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/analytics')
def analytics_page(): return render_template('analytics.html')

# --- CHATBOT BACKEND ROUTE ---
@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    data = request.json
    user_msg = data.get('message', '')
    context = data.get('context', '')
    
    prompt = f"{context} User asked: '{user_msg}'. Provide a very concise, empathetic wellness recommendation. Keep it short."
    
    try:
        response = gemini_model.generate_content(prompt)
        return jsonify({'reply': response.text})
    except Exception as e:
        print(f"Chatbot Error: {e}")
        return jsonify({'reply': "I'm having a little trouble connecting to my brain right now, but please remember to sit up straight and take a deep breath!"}), 500

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    try:
        data = request.json['image']
        img = cv2.imdecode(np.frombuffer(base64.b64decode(data.split(",")[1]), dtype=np.uint8), cv2.IMREAD_COLOR)
        return jsonify(analyze_multimodal(img))
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/predict_audio', methods=['POST'])
def predict_audio():
    try:
        audio_file = request.files['audio']
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            audio_file.save(temp_audio.name)
            y, sr = librosa.load(temp_audio.name, sr=None)
            energy = np.mean(librosa.feature.rms(y=y)) if len(y) > 0 else 0
            v_res = "Stress Detected" if energy > user_baseline['vocal_threshold'] else "Optimal"
            conn = get_db_connection()
            conn.execute("INSERT INTO posture_logs (timestamp, status, vocal_status, visual_status, drowsy_status, mode) VALUES (?, ?, ?, ?, ?, ?)",
                         (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Audio Log", v_res, "N/A", "N/A", "Vocal"))
            conn.commit(); conn.close(); os.remove(temp_audio.name)
            return jsonify({'status': 'success', 'vocal_stress': v_res})
    except: return jsonify({'status': 'error'})

@app.route('/get_history_all')
def get_history_all():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, status, visual_status, drowsy_status, vocal_status, mode FROM posture_logs ORDER BY id DESC LIMIT 10")
    rows = cursor.fetchall()
    conn.close()
    return jsonify([{'time': r[0], 'posture': r[1], 'visual': r[2], 'drowsy': r[3], 'vocal': r[4], 'mode': r[5]} for r in rows])

@app.route('/get_vocal_history')
def get_vocal_history():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, vocal_status FROM posture_logs WHERE mode = 'Vocal' ORDER BY id DESC LIMIT 10")
    rows = cursor.fetchall()
    conn.close()
    return jsonify([{'time': r[0], 'status': r[1]} for r in rows])

@app.route('/api/stress_data')
def get_stress_data():
    today = datetime.now().strftime("%Y-%m-%d")
    hours = list(range(24))
    vis, voc, drow, pos, over = [], [], [], [], []
    conn = get_db_connection()
    cursor = conn.cursor()
    for h in hours:
        prefix = f"{today} {h:02d}:%" 
        cursor.execute("SELECT status, vocal_status, visual_status, drowsy_status FROM posture_logs WHERE timestamp LIKE ?", (prefix,))
        rows = cursor.fetchall()
        if not rows:
            v_val = vc_val = p_val = d_val = 0
        else:
            total = len(rows)
            v_val = (sum(1 for r in rows if str(r[2]).strip().lower() == 'stressed') / total * 100)
            vc_val = (sum(1 for r in rows if str(r[1]).strip().lower() == 'stress detected') / total * 100)
            p_val = (sum(1 for r in rows if 'incorrect' in str(r[0]).lower()) / total * 100)
            d_val = (sum(1 for r in rows if 'drowsy' in str(r[3]).lower() or 'yawn' in str(r[3]).lower()) / total * 100)
        vis.append(round(v_val, 1)); voc.append(round(vc_val, 1)); drow.append(round(d_val, 1)); pos.append(round(p_val, 1))
        over.append(round((v_val * 0.4) + (vc_val * 0.1) + (d_val * 0.3) + (p_val * 0.2), 1))
    conn.close()
    best_h = over.index(min(over)) if any(over) else 0
    return jsonify({'labels': [f"{h:02d}:00" for h in hours], 'visual': vis, 'vocal': voc, 'drowsiness': drow, 'posture': pos, 'overall': over, 'best_hour': f"{best_h:02d}:00"})

@app.route('/calibrate', methods=['POST'])
def calibrate():
    try:
        data = request.json['image']
        img = cv2.imdecode(np.frombuffer(base64.b64decode(data.split(",")[1]), dtype=np.uint8), cv2.IMREAD_COLOR)
        results = pose_estimator.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            l = results.pose_landmarks.landmark
            
            is_front = abs(l[11].x - l[12].x) > 0.22
            
            if is_front:
                user_baseline['front_neck_dist'] = ((l[11].y + l[12].y) / 2) - l[0].y
            else:
                user_baseline['side_neck_angle'] = calculate_angle([l[7].x, l[7].y], [l[11].x, l[11].y], [l[23].x, l[23].y])
                user_baseline['side_back_angle'] = calculate_angle([l[11].x, l[11].y], [l[23].x, l[23].y], [l[25].x, l[25].y])
            user_baseline['is_calibrated'] = True
            save_baseline_to_disk()
            return jsonify({'status': 'success'})
    except: pass
    return jsonify({'status': 'error'})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)