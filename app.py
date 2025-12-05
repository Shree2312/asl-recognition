"""
==============================================================================
ASL SIGN LANGUAGE RECOGNITION - WEB APPLICATION
==============================================================================

A Flask-based web application with a modern UI for real-time ASL recognition.

Features:
- Real-time webcam feed with hand landmark detection
- Live letter prediction display
- Beautiful glassmorphism UI design
- Responsive and smooth animations

Author: AI Assistant
==============================================================================
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
from collections import Counter

app = Flask(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

MODEL_FILE = "knn_model.joblib"
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# Global variables
model = None
hands = None
mp_hands = None
mp_drawing = None
mp_drawing_styles = None
current_prediction = ""
prediction_history = []
HISTORY_SIZE = 5


def load_resources():
    """Load ML model and initialize MediaPipe."""
    global model, hands, mp_hands, mp_drawing, mp_drawing_styles
    
    # Load KNN model
    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
        print("✓ Model loaded successfully!")
    else:
        print("✗ Model not found. Please train the model first.")
        model = None
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )
    print("✓ MediaPipe initialized!")


def normalize_landmarks(hand_landmarks):
    """Normalize hand landmarks by subtracting wrist position."""
    wrist = hand_landmarks.landmark[0]
    wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z
    
    normalized_features = []
    for i in range(1, 21):
        landmark = hand_landmarks.landmark[i]
        normalized_features.extend([
            landmark.x - wrist_x,
            landmark.y - wrist_y,
            landmark.z - wrist_z
        ])
    
    return np.array(normalized_features).reshape(1, -1)


def generate_frames():
    """Generate video frames with predictions."""
    global current_prediction, prediction_history
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw landmarks with custom style
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=2)
            )
            
            # Make prediction
            if model is not None:
                features = normalize_landmarks(hand_landmarks)
                prediction = model.predict(features)[0]
                
                prediction_history.append(prediction)
                if len(prediction_history) > HISTORY_SIZE:
                    prediction_history.pop(0)
                
                if prediction_history:
                    current_prediction = Counter(prediction_history).most_common(1)[0][0]
        else:
            prediction_history.clear()
            current_prediction = ""
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/prediction')
def get_prediction():
    """Get current prediction."""
    return jsonify({'letter': current_prediction})


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("ASL SIGN LANGUAGE RECOGNITION - WEB APP")
    print("=" * 60 + "\n")
    
    load_resources()
    
    print("\n" + "-" * 60)
    print("Starting web server...")
    print("Open http://127.0.0.1:5000 in your browser")
    print("-" * 60 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
