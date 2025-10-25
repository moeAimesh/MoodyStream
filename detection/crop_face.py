import cv2
import mediapipe as mp

# Initialisiere Mediapipe FaceDetection nur einmal
_mp_face = mp.solutions.face_detection
_face_detector = _mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def crop_face(frame, draw_box=False): ##sorgt dafür dass deepface nur das gesicht bekommt und nicht den ganzen bildausschnitt 
    """
    Erkennt ein Gesicht im Frame und gibt den gecroppten Ausschnitt zurück.
    Optional: draw_box=True zeigt ein Rechteck im Originalbild an.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = _face_detector.process(rgb)

    if not result.detections:
        return None  # Kein Gesicht erkannt

    H, W, _ = frame.shape
    det = result.detections[0].location_data.relative_bounding_box
    x, y, w, h = det.xmin, det.ymin, det.width, det.height

    # Begrenzungen und Sicherheitsbereich
    x1, y1 = max(0, int(x * W)), max(0, int(y * H))
    x2, y2 = min(W, int((x + w) * W)), min(H, int((y + h) * H))

    if draw_box:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None  # Schutz gegen leere Ausschnitte
    return face_crop
