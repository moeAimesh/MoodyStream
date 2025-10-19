"""gibt True, wenn Pose Daumen-hoch ist.
Kann von gesture_recognition/emotion_recognition genutzt werden, um Entscheidungen modular zu halten.
""" 


import numpy as np

def finger_extended(landmarks, finger_indices, flipped):
    tip_y = landmarks[finger_indices[3]][1]
    mid_y = landmarks[finger_indices[2]][1]
    base_y = landmarks[finger_indices[1]][1]
    return tip_y > mid_y > base_y if flipped else tip_y < mid_y < base_y

def detect_thumbsup(hand_landmarks, w, h):
    """Gibt True zurück, wenn nur der Daumen gestreckt ist."""
    landmarks = np.array([(lm.x * w, lm.y * h, lm.z)
                          for lm in hand_landmarks.landmark])
    index_base = landmarks[5]
    pinky_base = landmarks[17]
    flipped = index_base[2] < pinky_base[2]

    fingers = {
        'thumb':  [1, 2, 3, 4],
        'index':  [5, 6, 7, 8],
        'middle': [9, 10, 11, 12],
        'ring':   [13, 14, 15, 16],
        'pinky':  [17, 18, 19, 20]
    }

    if flipped:
        thumb_up = landmarks[fingers['thumb'][3]][1] > landmarks[fingers['thumb'][2]][1]
    else:
        thumb_up = landmarks[fingers['thumb'][3]][1] < landmarks[fingers['thumb'][2]][1]

    index_up = finger_extended(landmarks, fingers['index'], flipped)
    middle_up = finger_extended(landmarks, fingers['middle'], flipped)
    ring_up = finger_extended(landmarks, fingers['ring'], flipped)
    pinky_up = finger_extended(landmarks, fingers['pinky'], flipped)

    return thumb_up and not (index_up or middle_up or ring_up or pinky_up)
