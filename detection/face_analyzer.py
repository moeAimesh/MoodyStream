"""Aufgabe: (passiert immer wenn programm läuft) Aus einem Frame Emotions-Scores holen (DeepFace) und ggf stabilisieren bzw mit Baseline vergleichen und passende Rechnungen durchführen.

Eingaben: Frame (BGR/RGB beachten).

Ausgaben: z. B. {"happy":0.73,"neutral":0.2,"sad":0.05,...}.

Performance-Hinweise: DeepFace-Modelle einmalig laden und cachen und idk seid offen und kreativ !!



API (Beispiel):
def analyze_face_emotions(frame: np.ndarray) -> dict:
    return {"happy":0.73,"neutral":0.2,"sad":0.05,...}
    """
