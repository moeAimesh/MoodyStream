import cv2
import mediapipe as mp
import numpy as np
from collections import defaultdict


GESTURES = {
    '1': 'thumbsup',
    '2': 'thumbsdown',
    '3': 'peace',
    '4': 'middlefinger',
    '5': 'open',
}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_normalized_landmarks(hand_landmarks):
    """
    Nimmt MediaPipe-Hand-Landmarks und gibt
    einen 63D-Vektor (21 Punkte x (x,y,z)) zurück,
    normalisiert relativ zum Handwurzel-Punkt (ID 0).
    """
    # Alle Punkte als (x, y, z)
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)

    # Handwurzel ist Punkt 0
    wrist = landmarks[0].copy()

    # Translation entfernen: alle Punkte - Handwurzel
    landmarks -= wrist  # Broadcasting

    # Optional: nochmal auf max. Distanz skalieren (Größenunterschiede noch mehr rausnehmen)


    # 21 x 3 -> 63 Features
    return landmarks.flatten()

def main():
    cap = cv2.VideoCapture(0)

    all_features = []
    all_labels = []

    counter = defaultdict(int)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Spiegeln für natürlicheres Feeling
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = hands.process(img_rgb)

            # Overlay-Text
            text_lines = [
                "Tasten: 1=thumbsup, 2=peace, 3=ok, 4=fist, 5=open",
                "Druecke q zum Beenden."
            ]
            y0 = 20
            for line in text_lines:
                cv2.putText(frame, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2, cv2.LINE_AA)
                y0 += 25

            # Zähler pro Geste anzeigen
            y0 += 10
            for key, name in GESTURES.items():
                cv2.putText(
                    frame,
                    f"{name}: {counter[name]}",
                    (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                y0 += 25

            # Hand zeichnen
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            cv2.imshow("Collect gesture data", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            # Wenn eine Geste-Taste gedrückt wird und eine Hand erkannt wurde
            if chr(key) in GESTURES and result.multi_hand_landmarks:
                gesture_name = GESTURES[chr(key)]
                # Nimm die erste Hand
                hand_landmarks = result.multi_hand_landmarks[0]
                features = extract_normalized_landmarks(hand_landmarks)

                all_features.append(features)
                all_labels.append(gesture_name)
                counter[gesture_name] += 1
                print(f"Gespeichert: {gesture_name} | Gesamt: {counter[gesture_name]}")

    cap.release()
    cv2.destroyAllWindows()

    # In NumPy-Arrays umwandeln
    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=object)

    print("Fertige Samples:", X.shape[0])
    print("Feature-Vektor Dimension:", X.shape[1])

    # Speichern
    np.savez("gesture_dataset.npz", X=X, y=y)

if __name__ == "__main__":
    main()
