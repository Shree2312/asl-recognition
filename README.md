# 🤟 ASL Sign Language Recognition System

A real-time American Sign Language (ASL) recognition system using webcam, MediaPipe, and K-Nearest Neighbors (KNN) machine learning.

## 📋 Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Tips for Best Results](#tips-for-best-results)

## ✨ Features

- **Real-time hand detection** using MediaPipe's 21-point hand landmark model
- **Position-invariant recognition** through wrist-based normalization
- **Keyboard-controlled data collection** for easy training data capture
- **KNN-based classification** for fast and accurate letter prediction
- **Prediction smoothing** using majority voting for stable results
- **Visual feedback** with hand skeleton overlay and prediction display

## 📦 Requirements

- Python 3.7+
- Webcam
- Required packages:
  - OpenCV (`opencv-python`)
  - MediaPipe (`mediapipe`)
  - Scikit-learn (`scikit-learn`)
  - NumPy (`numpy`)
  - Joblib (`joblib`)

## 🔧 Installation

1. **Clone or download this project**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Phase 1: Data Collection

Run the data collection script to capture hand gesture samples:

```bash
python data_collection.py
```

**Instructions:**
- Position your hand in front of the webcam
- Press letter keys (A-Z) to save samples for that letter
- Collect **at least 50-100 samples per letter** for good accuracy
- Try different hand positions and angles for each letter
- Press `Q` or `ESC` to exit

The data is saved to `alphabet_landmarks.csv`.

### Phase 2: Model Training

After collecting enough data, train the KNN model:

```bash
python train_model.py
```

This script will:
- Load your collected data
- Split it into training/testing sets
- Train a KNN classifier (k=5)
- Display accuracy metrics
- Save the model to `knn_model.joblib`

### Phase 3: Real-Time Prediction

Run the real-time detector to see your model in action:

```bash
python realtime_detector.py
```

**Instructions:**
- Show ASL hand signs to the webcam
- The predicted letter will be displayed on screen
- Press `Q` or `ESC` to exit

## 📁 Project Structure

```
sign lang/
├── data_collection.py    # Collect hand gesture training data
├── train_model.py        # Train the KNN classifier
├── realtime_detector.py  # Real-time prediction
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── alphabet_landmarks.csv # Generated training data
└── knn_model.joblib     # Trained model file
```

## 🔬 How It Works

### 1. Hand Landmark Detection
MediaPipe Hands detects 21 3D landmarks on the hand:
- Wrist (1 point)
- Thumb (4 points)
- Index finger (4 points)
- Middle finger (4 points)
- Ring finger (4 points)
- Pinky finger (4 points)

### 2. Normalization
To make the model position-invariant, all landmarks are normalized by subtracting the wrist position:
```
normalized_landmark = landmark - wrist_position
```

This creates 60 features (20 landmarks × 3 coordinates).

### 3. Classification
The K-Nearest Neighbors (KNN) algorithm classifies the normalized landmarks by finding the k=5 closest training samples and using majority voting.

### 4. Prediction Smoothing
To reduce flickering, the detector uses a sliding window of 5 predictions and displays the most common one.

## 💡 Tips for Best Results

1. **Lighting**: Ensure good, consistent lighting
2. **Background**: Use a plain, contrasting background
3. **Hand Position**: Keep your hand clearly visible and centered
4. **Data Variety**: Collect samples from different angles and distances
5. **Consistent Signs**: Follow standard ASL hand shapes
6. **Sample Size**: More samples = better accuracy (aim for 100+ per letter)

## 📊 Expected Accuracy

With proper data collection:
- **50 samples/letter**: ~85-90% accuracy
- **100+ samples/letter**: ~95%+ accuracy

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Webcam not detected | Check webcam connection and permissions |
| Low accuracy | Collect more training samples |
| Flickering predictions | Model is working correctly with smoothing |
| "Model not found" error | Run `train_model.py` first |

## 📝 License

This project is open source and available for educational purposes.

---

Made with ❤️ for the deaf and hard-of-hearing community
