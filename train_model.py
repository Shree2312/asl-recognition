"""
==============================================================================
MODEL TRAINING SCRIPT FOR SIGN LANGUAGE RECOGNITION
==============================================================================

This script trains a K-Nearest Neighbors (KNN) classifier using the
collected landmark data for American Sign Language (ASL) recognition.

Features:
- Loads normalized landmark features from CSV file
- Splits data into training and testing sets
- Trains a KNN classifier with configurable k value
- Evaluates model accuracy on test set
- Saves trained model using Joblib for later use

Author: AI Assistant
==============================================================================
"""

import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Input CSV file with collected landmark data
CSV_FILE = "alphabet_landmarks.csv"

# Output model file
MODEL_FILE = "knn_model.joblib"

# KNN configuration
K_NEIGHBORS = 5  # Number of neighbors for KNN

# Train-test split configuration
TEST_SIZE = 0.2  # 20% for testing
RANDOM_STATE = 42  # For reproducibility


def load_data(csv_file):
    """
    Load features and labels from the CSV file.
    
    Args:
        csv_file: Path to the CSV file containing landmark data
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the label array
    """
    print(f"Loading data from: {csv_file}")
    
    features = []
    labels = []
    
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        
        # Skip header row
        headers = next(reader)
        print(f"  Found {len(headers) - 1} feature columns + 1 label column")
        
        # Read data rows
        for row in reader:
            if len(row) == len(headers):
                # Extract features (all columns except last)
                feature_values = [float(val) for val in row[:-1]]
                features.append(feature_values)
                
                # Extract label (last column)
                label = row[-1]
                labels.append(label)
    
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    print(f"  Loaded {len(X)} samples")
    print(f"  Feature shape: {X.shape}")
    print(f"  Unique labels: {sorted(set(y))}")
    
    return X, y


def analyze_data(X, y):
    """
    Analyze the loaded data and print statistics.
    
    Args:
        X: Feature matrix
        y: Label array
    """
    print("\n" + "=" * 60)
    print("DATA ANALYSIS")
    print("=" * 60)
    
    # Count samples per class
    unique_labels, counts = np.unique(y, return_counts=True)
    
    print(f"\nTotal samples: {len(y)}")
    print(f"Number of classes: {len(unique_labels)}")
    print(f"Features per sample: {X.shape[1]}")
    
    print("\nSamples per class:")
    for label, count in zip(unique_labels, counts):
        bar = "█" * min(count, 50)
        print(f"  {label}: {count:4d} {bar}")
    
    print(f"\nMin samples per class: {min(counts)}")
    print(f"Max samples per class: {max(counts)}")
    print(f"Mean samples per class: {np.mean(counts):.1f}")
    
    # Warn if classes are imbalanced
    if max(counts) > 2 * min(counts):
        print("\n⚠ Warning: Class imbalance detected!")
        print("  Consider collecting more samples for underrepresented classes.")


def train_model(X_train, y_train, k=5):
    """
    Train a K-Nearest Neighbors classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        k: Number of neighbors to use
        
    Returns:
        KNeighborsClassifier: Trained KNN model
    """
    print(f"\nTraining KNN classifier with k={k}...")
    
    # Initialize KNN classifier
    knn = KNeighborsClassifier(
        n_neighbors=k,
        weights='uniform',  # All neighbors weighted equally
        algorithm='auto',   # Let sklearn choose best algorithm
        metric='euclidean'  # Euclidean distance for landmark coordinates
    )
    
    # Train the model
    knn.fit(X_train, y_train)
    
    print("✓ Model training complete!")
    
    return knn


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.
    
    Args:
        model: Trained KNN model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        float: Accuracy score
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✓ Test Accuracy: {accuracy * 100:.2f}%")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print("-" * 60)
    print(classification_report(y_test, y_pred))
    
    # Print confusion matrix for small number of classes
    unique_labels = sorted(set(y_test))
    if len(unique_labels) <= 10:
        print("Confusion Matrix:")
        print("-" * 60)
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        
        # Print header
        print("     ", end="")
        for label in unique_labels:
            print(f"{label:5}", end="")
        print()
        
        # Print matrix with row labels
        for i, label in enumerate(unique_labels):
            print(f"{label}:   ", end="")
            for j in range(len(unique_labels)):
                print(f"{cm[i, j]:5}", end="")
            print()
    
    return accuracy


def save_model(model, filepath):
    """
    Save the trained model to a file using Joblib.
    
    Args:
        model: Trained model to save
        filepath: Path to save the model
    """
    print(f"\nSaving model to: {filepath}")
    joblib.dump(model, filepath)
    
    # Verify the saved file
    file_size = os.path.getsize(filepath)
    print(f"✓ Model saved successfully! (Size: {file_size / 1024:.1f} KB)")


def main():
    """
    Main function to train and evaluate the KNN model.
    """
    print("=" * 60)
    print("ASL SIGN LANGUAGE MODEL TRAINING")
    print("=" * 60)
    
    # Check if CSV file exists
    if not os.path.exists(CSV_FILE):
        print(f"\n✗ Error: CSV file not found: {CSV_FILE}")
        print("  Please run data_collection.py first to collect training data.")
        return
    
    # Load data from CSV
    print("\n[1/5] Loading Data")
    print("-" * 40)
    X, y = load_data(CSV_FILE)
    
    # Check if we have enough data
    if len(X) < 10:
        print(f"\n✗ Error: Not enough samples ({len(X)} found)")
        print("  Please collect at least 10 samples before training.")
        return
    
    # Analyze the data
    print("\n[2/5] Analyzing Data")
    print("-" * 40)
    analyze_data(X, y)
    
    # Split data into training and testing sets
    print("\n[3/5] Splitting Data")
    print("-" * 40)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # Maintain class distribution in splits
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Test size: {TEST_SIZE * 100:.0f}%")
    
    # Train the model
    print("\n[4/5] Training Model")
    print("-" * 40)
    model = train_model(X_train, y_train, k=K_NEIGHBORS)
    
    # Evaluate the model
    print("\n[5/5] Evaluating Model")
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Save the model
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    save_model(model, MODEL_FILE)
    
    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n  Model: K-Nearest Neighbors (k={K_NEIGHBORS})")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test accuracy: {accuracy * 100:.2f}%")
    print(f"  Saved to: {MODEL_FILE}")
    print("\nYou can now run realtime_detector.py for live predictions!")
    print("=" * 60)


if __name__ == "__main__":
    main()
