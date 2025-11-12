import cv2
from deepface import DeepFace
from sklearn.neighbors import NearestNeighbors
import numpy as np
import json
import time
from pathlib import Path

from utils.settings import REST_FACE_MODEL_PATH


class RestFaceCalibrator:
    """
    Kalibriert das Rest-Face des Nutzers.
    Speichert DeepFace-Emotionsvektoren und trainiert daraus ein Nearest-Neighbour-Modell.
    """

    def __init__(self, model_path=REST_FACE_MODEL_PATH):
        self.model_path = Path(model_path)
        self.vectors = []
        self.model = None

    def record_rest_face(self, duration=20, analyze_every=5):
        """
        Nimmt mehrere Frames auf und extrahiert die Emotion-Vektoren des Rest-Face.
        """
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame_count = 0
        start = time.time()

        print("📷 Bitte mit neutralem Gesicht in die Kamera schauen...")

        while time.time() - start < duration:
            ret, frame = cam.read()
            if not ret:
                continue

            cv2.putText(frame, "Calibrating Rest Face...", (40, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Rest Face Calibration", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                print("❌ Abgebrochen")
                break

            if frame_count % analyze_every == 0:
                try:
                    result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
                    emotion_vec = np.array(list(result[0]["emotion"].values()))
                    self.vectors.append(emotion_vec)
                    print(f"🟢 Frame {len(self.vectors)} aufgenommen")
                except Exception as e:
                    print("⚠️ Analysefehler:", e)

            frame_count += 1

        cam.release()
        cv2.destroyAllWindows()

        if len(self.vectors) == 0:
            print("⚠️ Keine Emotionen erkannt.")
            return False

        print(f"✅ {len(self.vectors)} Rest-Face-Vektoren gesammelt.")
        return True

    def train(self):
        """
        Trainiert ein Nearest-Neighbor-Modell auf den Rest-Face-Vektoren.
        """
        if len(self.vectors) < 5:
            print("⚠️ Zu wenige Daten für Training!")
            return False

        X = np.array(self.vectors)
        self.model = NearestNeighbors(n_neighbors=1, metric="cosine")  # Cosine-Distanz macht mehr Sinn bei Emotionen als euklidische weil sie den Winkel zwischen Vektoren berücksichtigt
        self.model.fit(X)

        print("🧠 Modell trainiert.")
        return True

    def save_model(self):
        """
        Speichert alle Vektoren (und Mittelwert) in JSON.
        """
        if not self.vectors:
            print("⚠️ Keine Daten zum Speichern.")
            return

        model_data = {
            "vectors": np.array(self.vectors).tolist(),
            "mean_vector": np.mean(self.vectors, axis=0).tolist()
        }

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with self.model_path.open("w") as f:
            json.dump(model_data, f, indent=2)

        print(f"💾 Modell gespeichert unter: {self.model_path}")

    def visualize_space(self):
        """
        Optional: PCA-Projektion zur visuellen Kontrolle
        """
        try:
            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt
            X = np.array(self.vectors)
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(X)
            plt.scatter(reduced[:, 0], reduced[:, 1], color='blue', label="Rest Face Frames")
            plt.legend()
            plt.title("Rest Face Emotion Space (2D PCA)")
            plt.show()
        except Exception as e:
            print("⚠️ Visualisierung fehlgeschlagen:", e)


# -------------------------------------------------------
# 🔹 Hauptfunktion – kann direkt ausgeführt werden
# -------------------------------------------------------
if __name__ == "__main__":
    calibrator = RestFaceCalibrator(model_path=REST_FACE_MODEL_PATH)
    if calibrator.record_rest_face(duration=30, analyze_every=5):
        if calibrator.train():
            calibrator.save_model()
            calibrator.visualize_space()
