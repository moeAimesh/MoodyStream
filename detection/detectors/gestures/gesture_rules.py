import numpy as np

def finger_extended(landmarks, finger_indices, flipped):
    tip_y = landmarks[finger_indices[3]][1]
    mid_y = landmarks[finger_indices[2]][1]
    base_y = landmarks[finger_indices[1]][1]
    return tip_y > mid_y > base_y if flipped else tip_y < mid_y < base_y


# THUMBS UP
def detect_thumbsup(landmarks, flipped):
    fingers = {
        'thumb':  [1, 2, 3, 4],
        'index':  [5, 6, 7, 8],
        'middle': [9, 10, 11, 12],
        'ring':   [13, 14, 15, 16],
        'pinky':  [17, 18, 19, 20]
    }

    # Daumen hoch
    if flipped:
        thumb_up = landmarks[fingers['thumb'][3]][1] > landmarks[fingers['thumb'][2]][1]
    else:
        thumb_up = landmarks[fingers['thumb'][3]][1] < landmarks[fingers['thumb'][2]][1]

    index_up = finger_extended(landmarks, fingers['index'], flipped)
    middle_up = finger_extended(landmarks, fingers['middle'], flipped)
    ring_up = finger_extended(landmarks, fingers['ring'], flipped)
    pinky_up = finger_extended(landmarks, fingers['pinky'], flipped)

    return thumb_up and not (index_up or middle_up or ring_up or pinky_up)


# THUMBSDOWN 
def detect_thumbsdown(landmarks, flipped):
    fingers = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # Alle anderen Finger müssen gekrümmt sein
    all_curled = True
    for i in range(0, len(fingers), 4):
        tip = landmarks[fingers[i+3]][1]
        base = landmarks[fingers[i]][1]
        if tip < base:  
            all_curled = False

    # Daumen zeigt nach unten
    thumb_tip = landmarks[4][1]
    thumb_base = landmarks[2][1]

    return (thumb_tip > thumb_base) and all_curled


# PEACE 
def detect_peace(landmarks, flipped):
    index_up = finger_extended(landmarks, [5,6,7,8], flipped)
    middle_up = finger_extended(landmarks, [9,10,11,12], flipped)
    ring_up = finger_extended(landmarks, [13,14,15,16], flipped)
    pinky_up = finger_extended(landmarks, [17,18,19,20], flipped)
    
    return index_up and middle_up and not ring_up and not pinky_up


#OPEN HAND
def detect_open(landmarks, flipped):
    index_up = finger_extended(landmarks, [5,6,7,8], flipped)
    middle_up = finger_extended(landmarks, [9,10,11,12], flipped)
    ring_up = finger_extended(landmarks, [13,14,15,16], flipped)
    pinky_up = finger_extended(landmarks, [17,18,19,20], flipped)

    # Alle Finger gestreckt
    return index_up and middle_up and ring_up and pinky_up


# mittelfinger
def detect_middlefinger(landmarks, flipped):
    index_up = finger_extended(landmarks, [5,6,7,8], flipped)
    middle_up = finger_extended(landmarks, [9,10,11,12], flipped)
    ring_up = finger_extended(landmarks, [13,14,15,16], flipped)
    pinky_up = finger_extended(landmarks, [17,18,19,20], flipped)

    return (not index_up) and middle_up and (not ring_up) and (not pinky_up)

def validate_gesture(label, hand_landmarks, w, h):
    # Landmark-Array
    landmarks = np.array([(lm.x * w, lm.y * h, lm.z) for lm in hand_landmarks.landmark])

    # Orientierung (links / rechts)
    index_base = landmarks[5]
    pinky_base = landmarks[17]
    flipped = index_base[2] < pinky_base[2]

    if label == "thumbsup":
        return detect_thumbsup(landmarks, flipped)
    if label == "thumbsdown":
        return detect_thumbsdown(landmarks, flipped)
    if label == "peace":
        return detect_peace(landmarks, flipped)
    if label == "open":
        return detect_open(landmarks, flipped)
    if label == "middlefinger":
        return detect_middlefinger(landmarks, flipped)

    return False
