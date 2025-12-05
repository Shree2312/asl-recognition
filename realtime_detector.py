"""
==============================================================================
REAL-TIME SIGN LANGUAGE DETECTION SCRIPT
==============================================================================

This script performs real-time American Sign Language (ASL) letter prediction
using a pre-trained KNN model and MediaPipe hand landmark detection.

Features:
- Loads pre-trained KNN model from Joblib file
- Uses MediaPipe Hands for 21-point hand landmark detection
- Applies the same wrist-based normalization as data collection
- Displays predicted letter prominently on video feed
- Draws hand landmarks and skeleton on frame

Author: AI Assistant
==============================================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import os


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Path to the saved KNN model
MODEL_FILE = "knn_model.joblib"

# MediaPipe Hands configuration
MAX_NUM_HANDS = 1  # Detect only one hand for simplicity
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# Webcam configuration
WEBCAM_INDEX = 0  # Default webcam
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Display configuration
FONT = cv2.FONT_HERSHEY_SIMPLEX
PREDICTION_FONT_SCALE = 2.0
PREDICTION_FONT_THICKNESS = 3


def normalize_landmarks(hand_landmarks):
    """
    Normalize hand landmarks by subtracting the wrist position.
    
    This is the EXACT SAME normalization method used in data_collection.py
    to ensure consistency between training and prediction.
    
    The wrist (landmark index 0) becomes the origin (0, 0, 0), making the
    model invariant to the hand's position in the frame.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks object containing 21 landmarks
        
    Returns:
        numpy.ndarray: Array of 60 normalized features (20 landmarks × 3 coordinates)
    """
    # Extract wrist coordinates (landmark index 0)
    wrist = hand_landmarks.landmark[0]
    wrist_x = wrist.x
    wrist_y = wrist.y
    wrist_z = wrist.z
    
    # Normalize all other landmarks by subtracting wrist position
    normalized_features = []
    
    for i in range(1, 21):  # Landmarks 1 to 20 (skip wrist at index 0)
        landmark = hand_landmarks.landmark[i]
        
        # Subtract wrist coordinates to normalize
        norm_x = landmark.x - wrist_x
        norm_y = landmark.y - wrist_y
        norm_z = landmark.z - wrist_z
        
        normalized_features.extend([norm_x, norm_y, norm_z])
    
    # Return as numpy array for sklearn prediction
    return np.array(normalized_features).reshape(1, -1)


def draw_prediction(frame, prediction, confidence=None):
    """
    Draw the predicted letter prominently on the frame.
    
    Args:
        frame: The OpenCV frame to draw on
        prediction: The predicted letter
        confidence: Optional confidence score (not used with basic KNN)
    """
    height, width = frame.shape[:2]
    
    # Draw a large box for the prediction at top-right
    box_width = 150
    box_height = 100
    box_x = width - box_width - 20
    box_y = 20
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), 
                  (box_x + box_width, box_y + box_height), 
                  (0, 100, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw border
    cv2.rectangle(frame, (box_x, box_y), 
                  (box_x + box_width, box_y + box_height), 
                  (0, 255, 0), 2)
    
    # Draw "Predicted:" label
    cv2.putText(frame, "Predicted:", (box_x + 10, box_y + 25), 
                FONT, 0.5, (255, 255, 255), 1)
    
    # Draw the predicted letter (large and centered)
    text_size = cv2.getTextSize(prediction, FONT, PREDICTION_FONT_SCALE, 
                                PREDICTION_FONT_THICKNESS)[0]
    text_x = box_x + (box_width - text_size[0]) // 2
    text_y = box_y + 75
    
    cv2.putText(frame, prediction, (text_x, text_y), 
                FONT, PREDICTION_FONT_SCALE, (0, 255, 0), 
                PREDICTION_FONT_THICKNESS)


def draw_status_panel(frame, hand_detected, model_loaded):
    """
    Draw a status panel showing system state.
    
    Args:
        frame: The OpenCV frame to draw on
        hand_detected: Boolean indicating if hand is detected
        model_loaded: Boolean indicating if model is loaded
    """
    height, width = frame.shape[:2]
    
    # Draw semi-transparent background for status panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (250, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Title
    cv2.putText(frame, "ASL Real-Time Detector", (20, 35), 
                FONT, 0.6, (255, 255, 255), 1)
    
    # Model status
    model_color = (0, 255, 0) if model_loaded else (0, 0, 255)
    model_text = "Model: LOADED" if model_loaded else "Model: NOT LOADED"
    cv2.putText(frame, model_text, (20, 55), 
                FONT, 0.5, model_color, 1)
    
    # Hand detection status
    hand_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    hand_text = "Hand: DETECTED" if hand_detected else "Hand: NOT DETECTED"
    cv2.putText(frame, hand_text, (20, 75), 
                FONT, 0.5, hand_color, 1)
    
    # Instructions at bottom
    cv2.putText(frame, "Press Q or ESC to quit", (20, height - 20), 
                FONT, 0.5, (150, 150, 150), 1)


def load_model(model_path):
    """
    Load the trained KNN model from a Joblib file.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        The loaded model, or None if loading fails
    """
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"✗ Error: Model file not found!")
        print(f"  Please run train_model.py first to train the model.")
        return None
    
    try:
        model = joblib.load(model_path)
        print("✓ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None


def main():
    """
    Main function to run the real-time sign language detector.
    """
    print("=" * 60)
    print("ASL REAL-TIME SIGN LANGUAGE DETECTOR")
    print("=" * 60)
    
    # Load the trained KNN model
    print("\n[1/3] Loading Model")
    print("-" * 40)
    model = load_model(MODEL_FILE)
    model_loaded = model is not None
    
    if not model_loaded:
        print("\n⚠ Running in demo mode (no predictions)")
        print("  To enable predictions, run train_model.py first.")
    
    # Initialize MediaPipe Hands
    print("\n[2/3] Initializing MediaPipe Hands")
    print("-" * 40)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )
    
    print("✓ MediaPipe Hands initialized")
    
    # Initialize webcam
    print("\n[3/3] Initializing Webcam")
    print("-" * 40)
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    if not cap.isOpened():
        print("✗ Error: Cannot open webcam!")
        hands.close()
        return
    
    print("✓ Webcam initialized")
    
    print("\n" + "=" * 60)
    print("Starting real-time detection... (press Q or ESC to quit)")
    print("=" * 60 + "\n")
    
    # Variables for smoothing predictions (optional)
    prediction_history = []
    HISTORY_SIZE = 5  # Number of frames to average predictions
    
    try:
        while True:
            # Capture frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("✗ Error: Cannot read frame from webcam!")
                break
            
            # Flip frame horizontally for mirror effect (more intuitive)
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe Hands
            results = hands.process(rgb_frame)
            
            # Check if hand is detected
            hand_detected = results.multi_hand_landmarks is not None
            current_prediction = None
            
            if hand_detected:
                # Get the first hand's landmarks
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Make prediction if model is loaded
                if model_loaded:
                    # Normalize landmarks (same method as data collection)
                    normalized_features = normalize_landmarks(hand_landmarks)
                    
                    # Predict the letter
                    prediction = model.predict(normalized_features)[0]
                    current_prediction = prediction
                    
                    # Add to prediction history for smoothing
                    prediction_history.append(prediction)
                    if len(prediction_history) > HISTORY_SIZE:
                        prediction_history.pop(0)
                    
                    # Use the most common prediction (majority voting)
                    if prediction_history:
                        from collections import Counter
                        most_common = Counter(prediction_history).most_common(1)[0][0]
                        current_prediction = most_common
            else:
                # Clear prediction history when no hand detected
                prediction_history.clear()
            
            # Draw the prediction on the frame
            if current_prediction:
                draw_prediction(frame, current_prediction)
            
            # Draw status panel
            draw_status_panel(frame, hand_detected, model_loaded)
            
            # Display the frame
            cv2.imshow("ASL Real-Time Detector", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            # Check for quit keys
            if key == ord('q') or key == ord('Q') or key == 27:  # 27 = ESC
                print("\n✓ Exiting real-time detector...")
                break
    
    finally:
        # Cleanup resources
        print("\nCleaning up resources...")
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        
        print("✓ Webcam released")
        print("✓ MediaPipe closed")
        print("✓ Windows destroyed")
        
        print("\n" + "=" * 60)
        print("SESSION ENDED")
        print("=" * 60)


if __name__ == "__main__":
    main()
