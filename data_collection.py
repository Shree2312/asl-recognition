"""
==============================================================================
DATA COLLECTION SCRIPT FOR SIGN LANGUAGE RECOGNITION
==============================================================================

This script captures American Sign Language (ASL) hand gestures via webcam
and saves the normalized landmark features to a CSV file for model training.

Features:
- Uses MediaPipe Hands for 21-point hand landmark detection
- Normalizes landmarks by subtracting wrist position (position-invariant)
- Saves 60 features (20 landmarks × 3 coordinates) per sample
- Press A-Z keys to save samples for each letter
- Press 'Q' or 'ESC' to exit

Author: AI Assistant
==============================================================================
"""

import cv2
import mediapipe as mp
import csv
import os
import numpy as np

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Path to the CSV file where landmark data will be stored
CSV_FILE = "alphabet_landmarks.csv"

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
FONT_SCALE = 1.0
FONT_THICKNESS = 2


def create_csv_with_headers(filepath):
    """
    Create a new CSV file with appropriate headers.
    
    Headers include:
    - 60 landmark features (x1, y1, z1, x2, y2, z2, ..., x20, y20, z20)
    - 1 label column
    
    Note: We have 20 normalized landmarks (landmarks 1-20, after subtracting wrist)
    """
    headers = []
    
    # Generate feature column names for landmarks 1-20 (excluding wrist at index 0)
    for i in range(1, 21):  # Landmarks 1 to 20
        headers.extend([f"x{i}", f"y{i}", f"z{i}"])
    
    # Add the label column
    headers.append("label")
    
    # Write headers to CSV file
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
    
    print(f"✓ Created new CSV file: {filepath}")
    print(f"  Headers: {len(headers) - 1} features + 1 label = {len(headers)} columns")


def normalize_landmarks(hand_landmarks):
    """
    Normalize hand landmarks by subtracting the wrist position.
    
    This makes the model invariant to the hand's position in the frame.
    The wrist (landmark index 0) becomes the origin (0, 0, 0).
    
    Args:
        hand_landmarks: MediaPipe hand landmarks object containing 21 landmarks
        
    Returns:
        list: Flattened list of 60 normalized features (20 landmarks × 3 coordinates)
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
    
    return normalized_features


def draw_info_panel(frame, current_letter, sample_count, hand_detected):
    """
    Draw an information panel on the frame showing current state.
    
    Args:
        frame: The OpenCV frame to draw on
        current_letter: The letter currently selected for collection (or None)
        sample_count: Number of samples collected for current letter
        hand_detected: Boolean indicating if a hand is detected
    """
    height, width = frame.shape[:2]
    
    # Draw semi-transparent background for info panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Title
    cv2.putText(frame, "ASL Data Collection", (20, 40), 
                FONT, 0.8, (255, 255, 255), 2)
    
    # Hand detection status
    status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    status_text = "Hand: DETECTED" if hand_detected else "Hand: NOT DETECTED"
    cv2.putText(frame, status_text, (20, 70), 
                FONT, 0.6, status_color, 2)
    
    # Current letter being collected
    if current_letter:
        cv2.putText(frame, f"Collecting: '{current_letter}'", (20, 100), 
                    FONT, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Samples: {sample_count}", (20, 130), 
                    FONT, 0.6, (255, 255, 255), 1)
    else:
        cv2.putText(frame, "Press A-Z to collect", (20, 100), 
                    FONT, 0.6, (200, 200, 200), 1)
    
    # Instructions at the bottom
    cv2.putText(frame, "Press Q or ESC to quit", (20, height - 20), 
                FONT, 0.5, (150, 150, 150), 1)


def main():
    """
    Main function to run the data collection script.
    """
    print("=" * 60)
    print("ASL DATA COLLECTION SCRIPT")
    print("=" * 60)
    print("\nInstructions:")
    print("  - Press A-Z keys to save samples for each letter")
    print("  - Make sure your hand is visible and detected")
    print("  - Collect multiple samples per letter for better accuracy")
    print("  - Press 'Q' or 'ESC' to exit\n")
    
    # Check if CSV file exists, if not create with headers
    if not os.path.exists(CSV_FILE):
        create_csv_with_headers(CSV_FILE)
    else:
        print(f"✓ Using existing CSV file: {CSV_FILE}")
    
    # Initialize MediaPipe Hands
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
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    if not cap.isOpened():
        print("✗ Error: Cannot open webcam!")
        return
    
    print("✓ Webcam initialized")
    print("\n" + "-" * 60)
    print("Starting data collection... (press Q or ESC to quit)")
    print("-" * 60 + "\n")
    
    # Track samples collected per letter
    sample_counts = {chr(i): 0 for i in range(ord('A'), ord('Z') + 1)}
    current_letter = None
    
    # Open CSV file for appending
    csv_file = open(CSV_FILE, mode='a', newline='')
    csv_writer = csv.writer(csv_file)
    
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
            normalized_features = None
            
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
                
                # Normalize landmarks (subtract wrist position)
                normalized_features = normalize_landmarks(hand_landmarks)
            
            # Draw information panel
            draw_info_panel(frame, current_letter, 
                          sample_counts.get(current_letter, 0) if current_letter else 0,
                          hand_detected)
            
            # Display the frame
            cv2.imshow("ASL Data Collection", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            # Check for quit key (ESC only - so 'Q' can be used for the letter Q)
            if key == 27:  # 27 = ESC
                print("\n✓ Exiting data collection...")
                break
            
            # Check for letter keys (A-Z)
            if key >= ord('a') and key <= ord('z'):
                key = key - 32  # Convert to uppercase
            
            if key >= ord('A') and key <= ord('Z'):
                letter = chr(key)
                current_letter = letter
                
                if hand_detected and normalized_features is not None:
                    # Save the normalized features with the label
                    row = normalized_features + [letter]
                    csv_writer.writerow(row)
                    csv_file.flush()  # Ensure data is written immediately
                    
                    sample_counts[letter] += 1
                    print(f"  ✓ Saved sample for '{letter}' "
                          f"(Total: {sample_counts[letter]})")
                else:
                    print(f"  ✗ No hand detected! Cannot save sample for '{letter}'")
    
    finally:
        # Cleanup resources
        csv_file.close()
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        
        # Print summary
        print("\n" + "=" * 60)
        print("DATA COLLECTION SUMMARY")
        print("=" * 60)
        
        total_samples = sum(sample_counts.values())
        print(f"\nTotal samples collected: {total_samples}")
        
        if total_samples > 0:
            print("\nSamples per letter:")
            for letter, count in sorted(sample_counts.items()):
                if count > 0:
                    print(f"  {letter}: {count} samples")
        
        print(f"\nData saved to: {CSV_FILE}")
        print("=" * 60)


if __name__ == "__main__":
    main()
