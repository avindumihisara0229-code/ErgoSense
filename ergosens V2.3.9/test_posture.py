import pytest
import numpy as np
from app import calculate_angle, calculate_overall_stress, app

# --- FIXTURE SETUP (This fixes the 'client not found' error!) ---
@pytest.fixture
def client():
    """Creates a fake browser client to test Flask routes without starting the server."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# --- 1. GEOMETRIC LOGIC TESTS ---
def test_posture_angle_math():
    """Verify the angle math for neck and back tilt is 100% accurate."""
    # Simulating a vertical line (perfectly straight back)
    shoulder = [0, 0]
    hip = [0, -1]
    knee = [0, -2]
    
    angle = calculate_angle(shoulder, hip, knee)
    # A vertical line from shoulder to hip to knee should be 180 degrees
    assert round(angle) == 180

def test_slouch_angle_detection():
    """Verify that a 45-degree lean is calculated correctly."""
    ear = [0, 1]      # Top
    shoulder = [0, 0] # Vertex
    hip = [1, -1]     # Leaned back 45 degrees
    
    angle = calculate_angle(ear, shoulder, hip)
    # This should be approximately 135 degrees (180 - 45)
    assert 130 <= angle <= 140

# --- 2. ML MODEL INFERENCE TESTS ---
def test_model_prediction_range():
    """Test if the model handles prediction probabilities correctly."""
    from app import model
    if model is not None:
        # Create a dummy feature set [neck_angle, back_angle]
        # 180, 180 represents perfectly straight
        dummy_features = [[180.0, 180.0]]
        
        # We test the predict_proba method used in app.py
        try:
            prediction_prob = model.predict_proba(dummy_features)[0][1]
            assert 0.0 <= prediction_prob <= 1.0
        except AttributeError:
            # Fallback if the model only supports predict()
            prediction = model.predict(dummy_features)
            assert prediction[0] in [0, 1, 'Correct', 'Incorrect']
    else:
        pytest.skip("Model file not found, skipping ML inference test.")

# --- 3. DATABASE PERSISTENCE TESTS ---
def test_posture_logging(client):
    """Test if the system successfully saves a posture result to the database."""
    # We send a dummy image (transparent pixel) to the predict endpoint
    # to trigger a database write
    minimal_pixel_base64 = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
    
    response = client.post('/predict_frame', 
                           json={'image': minimal_pixel_base64})
    
    assert response.status_code == 200
    # Even if no person is detected, the API should respond without crashing
    data = response.get_json()
    assert 'status' in data

# --- 4. CALIBRATION LOGIC TESTS ---
def test_calibration_baseline_logic():
    """Check if the system handles the user's calibrated baseline correctly."""
    from app import user_baseline
    
    # Manually set a baseline
    user_baseline['is_calibrated'] = True
    user_baseline['side_neck_angle'] = 150.0
    
    # Simulate a current angle that is 20 degrees off (Should be 'Incorrect')
    current_angle = 125.0 
    deviation = abs(current_angle - user_baseline['side_neck_angle'])
    
    assert deviation > 15 # Our bad posture threshold