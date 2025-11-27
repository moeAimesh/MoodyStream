import numpy as np
import joblib

model = joblib.load("svm_gesture_model.pkl")

def extract_features(hand_landmarks):
    points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
    wrist = points[0]
    points -= wrist
    return points.flatten().reshape(1, -1)

def classify_gesture(hand_landmarks):
    X = extract_features(hand_landmarks)
    return model.predict(X)[0]
