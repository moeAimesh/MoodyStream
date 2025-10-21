import numpy as np
from sklearn.neighbors import NearestNeighbors
import json
import os

class RestFaceFilter:
    """
    Filtert Emotionserkennungen basierend auf dem Rest-Face des Nutzers.
    """

    def __init__(self, model_path="setup/rest_face_model.json"):
        self.model_path = model_path
        self.rest_face_vectors = []
        self.nbrs = None

        # Modell laden, falls schon vorhanden
        if os.path.exists(self.model_path):
            self.load_model()

    def add_vector(self, emotion_scores: dict):
        """
        Fügt einen neuen DeepFace-Emotion-Vektor hinzu (z. B. während Kalibrierung).
        """
        vector = np.array(list(emotion_scores.values()), dtype=float)
        self.rest_face_vectors.append(vector)

    def train_model(self):
        """
        Trainiert einen NearestNeighbour auf Basis aller gespeicherten Rest-Face-Vektoren.
        """
        if len(self.rest_face_vectors) < 5:
            print("⚠️ Zu wenige Samples für Rest-Face-Modell.")
            return

        self.nbrs = NearestNeighbors(n_neighbors=1, metric="euclidean")
        self.nbrs.fit(self.rest_face_vectors)

    def save_model(self):
        """
        Speichert das trainierte Modell (Mittelwerte + Vektoren).
        """
        data = np.array(self.rest_face_vectors).tolist()
        with open(self.model_path, "w") as f:
            json.dump(data, f)
        print(f"💾 Rest-Face-Modell gespeichert unter: {self.model_path}")

    def load_model(self):
        """
        Lädt das gespeicherte Modell.
        """
        with open(self.model_path, "r") as f:
            data = json.load(f)
        self.rest_face_vectors = [np.array(v) for v in data]
        self.train_model()
        print(f"✅ Rest-Face-Modell geladen ({len(self.rest_face_vectors)} Samples).")

    def is_rest_face(self, emotion_scores: dict, threshold=0.12) -> bool:
        """
        Prüft, ob der aktuelle Frame wahrscheinlich das Rest-Face ist.
        threshold: Distanz-Schwelle (je kleiner, desto strenger)
        """
        if not self.nbrs:
            print("⚠️ Kein Modell geladen/trainiert.")
            return False

        vector = np.array(list(emotion_scores.values()), dtype=float)
        distance, _ = self.nbrs.kneighbors([vector])
        return distance[0][0] < threshold
