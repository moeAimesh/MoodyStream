import base64
from io import BytesIO

import cv2
from deepface import DeepFace
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from numpy.random import default_rng

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection.face_detection import detect_faces
from detection.preprocessing import preprocess_face
from detection.facemesh_features import extract_facemesh_features
from utils.settings import REST_FACE_MODEL_PATH, TRAINED_MODEL

EMOTION_PROFILES: List[Tuple[str, str]] = [
    ("fear", "Bitte ?ngstlich erstaunt schauen (Augen aufrei?en, Mund leicht ge?ffnet)."),
    ("happy", "Bitte lachen / fr?hlich schauen (breites L?cheln)."),
    ("sad", "Bitte traurig oder betr?bt schauen (Mundwinkel nach unten, Blick senken)."),
    ("surprise", "Bitte ?berrascht schauen (Augenbrauen hoch, Mund zu einem 'O')."),
    ("neutral", "Bitte entspannt / neutral schauen."),
]
class RestFaceCalibrator:
    """
    Kalibriert mehrere Emotionen des Nutzers.
    Speichert pro Emotion die DeepFace-Embeddings + Statistiken.
    """

    def __init__(self, model_path=REST_FACE_MODEL_PATH, snapshot_path=None):
        self.model_path = Path(model_path)
        self.snapshot_path = (
            Path(snapshot_path)
            if snapshot_path
            else self.model_path.with_name(f"{self.model_path.stem}.profiles.snapshot.json")
        )
        self.profiles = {}
        self.neutral_nn = None
        self.classifier_params = None
        self.scaler_params = None
        self.rng = default_rng()
        self.classifier_model_type = None
        self.neutral_feature_mean = None
        self.computed_thresholds = {"gate": 0.60, "margin": 0.08}
        self.neutral_feature_mean = None

    def record_emotions(
        self,
        emotions: Optional[Sequence[str]] = None,
        duration: int = 10,
        analyze_every: int = 5,
    ):
        """
        Führt den Nutzer durch verschiedene Emotionen und sammelt Embeddings.
        Optional können einzelne Emotionen übergeben werden, um gezielt Profile zu aktualisieren.
        """
        try:
            targets = self._resolve_emotions(emotions)
        except ValueError as exc:
            print(f"⚠️ {exc}")
            return False
        if not targets:
            print("⚠️ Keine Emotionen ausgewählt.")
            return False

        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Warm-up: Modelle/Kamera initialisieren, bevor Samples gezählt werden.
        self._warmup_session(cam)

        reset_profiles = emotions is None
        if reset_profiles:
            self.profiles = {}

        try:
            for emotion, instruction in targets:
                payload = self._capture_emotion_session(
                    cam,
                    emotion,
                    instruction,
                    duration=duration,
                    analyze_every=analyze_every,
                )
                if payload is None:
                    return False
                self.profiles[emotion] = payload
        finally:
            cam.release()
            cv2.destroyAllWindows()

        self.neutral_feature_mean = self._compute_neutral_feature_mean()
        self._save_snapshot()
        if emotions is None:
            print("�o. Alle Emotionen erfolgreich erfasst.")
        else:
            joined = ", ".join(e for e, _ in targets)
            print(f"✅ Emotion(en) aktualisiert: {joined}")
        return True

    def record_single_emotion(self, emotion: str, duration: int = 10, analyze_every: int = 5) -> bool:
        """Convenience-Methode, um genau eine Emotion aufzunehmen."""
        return self.record_emotions(emotions=[emotion], duration=duration, analyze_every=analyze_every)

    def _resolve_emotions(self, emotions: Optional[Sequence[str]]) -> List[Tuple[str, str]]:
        if emotions is None:
            return list(EMOTION_PROFILES)
        resolved: List[Tuple[str, str]] = []
        seen = set()
        for raw in emotions:
            if raw is None:
                continue
            name = raw.strip().lower()
            match = next((item for item in EMOTION_PROFILES if item[0].lower() == name), None)
            if match is None:
                raise ValueError(f"Unbekannte Emotion: {raw}")
            if match[0] in seen:
                continue
            resolved.append(match)
            seen.add(match[0])
        return resolved

    def _capture_emotion_session(
        self,
        cam,
        emotion: str,
        instruction: str,
        *,
        duration: int,
        analyze_every: int,
    ):
        if not self._wait_for_start(cam, emotion, instruction):
            print("?? Abgebrochen durch Nutzer.")
            return None

        vectors = []
        feature_vectors = []
        sample_crops = []
        frame_count = 0
        start = time.time()
        print(f"\n�Y\"� Profil '{emotion}' – Aufnahme gestartet")

        while time.time() - start < duration:
            ret, frame = cam.read()
            if not ret:
                continue

            boxes = detect_faces(frame)
            if boxes:
                x, y, w, h = boxes[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"Profil: {emotion}",
                (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )
            cv2.imshow("Emotion Profiling", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                print("�?O Abgebrochen")
                return None

            if frame_count % analyze_every == 0 and boxes:
                processed = preprocess_face(frame, boxes[0])
                if processed is None:
                    frame_count += 1
                    continue
                features = extract_facemesh_features(frame, boxes[0])
                if features is None:
                    frame_count += 1
                    continue
                try:
                    result = DeepFace.analyze(
                        processed,
                        actions=["emotion"],
                        enforce_detection=False,
                        detector_backend="mediapipe",
                    )
                    emotion_vec = np.array(list(result[0]["emotion"].values()))
                    vectors.append(emotion_vec)
                    feature_vectors.append(features)
                    if len(sample_crops) < 3:
                        sample_crops.append(processed)
                    print(f"�YY� {emotion}: Sample {len(vectors)}")
                except Exception as exc:
                    print("�s���? Analysefehler:", exc)

            frame_count += 1

        if len(vectors) < 3:
            print(f"�s���? Zu wenige Samples für {emotion}. Bitte erneut versuchen.")
            return None

        return {
            "vectors": vectors,
            "features": feature_vectors,
            "samples": sample_crops,
        }

    def train(self):
        """
        Optional: trainiert ein NN-Modell auf den neutralen Vektoren.
        """
        neutral_vectors = self.profiles.get("neutral", {}).get("vectors")
        if neutral_vectors and len(neutral_vectors) >= 5:
            X = np.array(neutral_vectors)
            self.neutral_nn = NearestNeighbors(n_neighbors=1, metric="cosine")
            self.neutral_nn.fit(X)
            print("✅✅ Neutral-Modell trainiert.")
        else:
            print("⚠️⚠️ Neutral-Profil zu klein – überspringe NN-Training.")

        clf_dataset = self._prepare_classifier_dataset()
        if clf_dataset is None:
            print("⚠️⚠️ Zu wenige Daten für Klassifikator.")
        else:
            X_balanced, y_balanced, feature_len = clf_dataset
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_balanced)
            model_type, clf = self._init_classifier(TRAINED_MODEL)
            clf.fit(X_scaled, y_balanced)
            self.classifier_params = self._build_classifier_payload(
                clf, model_type, feature_len, scaler
            )
            self.classifier_model_type = model_type
            print("✅ Personalisierter Klassifikator trainiert.")
        # Dynamic thresholds from neutral noise
        self._compute_thresholds_from_neutral()
        return True
    def save_model(self):
        """
        Speichert pro Emotion Mittelwerte, Distanzen und Beispielbilder.
        """
        if not self.profiles:
            print("�s���? Keine Daten zum Speichern.")
            return

        model_data = {"profiles": {}}
        for emotion, payload in self.profiles.items():
            vectors_np, mean_vec, distances, stats = self._compute_stats(payload["vectors"])
            samples = self._save_samples(emotion, payload["samples"])

            model_data["profiles"][emotion] = {
                "vectors": vectors_np.tolist(),
                "feature_vectors": np.array(payload["features"]).tolist(),
                "mean_vector": mean_vec.tolist(),
                "distance_mean": stats["mean"],
                "distance_std": stats["std"],
                "distance_min": stats["min"],
                "distance_max": stats["max"],
                "sample_images": samples,
            }
        if self.classifier_params:
            model_data["classifier"] = self.classifier_params
        if self.neutral_feature_mean is not None:
            model_data["neutral_feature_mean"] = self.neutral_feature_mean.tolist()

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with self.model_path.open("w", encoding="utf-8") as f:
            json.dump(model_data, f, indent=2)

        print(f"�Y'� Profil gespeichert unter: {self.model_path}")

    def visualize_space(self):
        """
        Optional: PCA-Projektion für alle Emotionen.
        """
        try:
            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt

            for emotion, payload in self.profiles.items():
                X = np.array(payload["vectors"])
                if X.shape[0] < 2:
                    continue
                pca = PCA(n_components=2)
                reduced = pca.fit_transform(X)
                plt.scatter(reduced[:, 0], reduced[:, 1], label=emotion)
            plt.legend()
            plt.title("Emotion Embedding Space (2D PCA)")
            plt.show()
        except Exception as exc:
            print("�s���? Visualisierung fehlgeschlagen:", exc)

    def _compute_stats(self, vectors):
        vectors_np = np.array(vectors)
        mean_vector = np.mean(vectors_np, axis=0)
        distances = np.linalg.norm(vectors_np - mean_vector, axis=1)
        stats = {
            "mean": float(np.mean(distances)),
            "std": float(np.std(distances) or 1e-6),
            "min": float(np.min(distances)),
            "max": float(np.max(distances)),
        }
        return vectors_np, mean_vector, distances, stats

    def _save_samples(self, emotion, crops):
        if not crops:
            return []
        samples_dir = self.model_path.parent / "samples" / emotion
        samples_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for idx, crop in enumerate(crops):
            path = samples_dir / f"{emotion}_sample_{idx}.png"
            cv2.imwrite(str(path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            saved.append(str(path.relative_to(self.model_path.parent)))
        return saved

    def _compute_neutral_feature_mean(self):
        """Return mean AU feature vector for neutral, if available."""
        neutral = self.profiles.get("neutral")
        if not neutral:
            return None
        feats = neutral.get("features") or []
        if not feats:
            return None
        feats_np = np.array(feats, dtype=float)
        if feats_np.size == 0:
            return None
        return feats_np.mean(axis=0)

    def _center_features(self, feats_np: np.ndarray) -> np.ndarray:
        if self.neutral_feature_mean is None:
            return feats_np
        if feats_np.shape[1] != self.neutral_feature_mean.shape[0]:
            return feats_np
        return feats_np - self.neutral_feature_mean

    def _compute_thresholds_from_neutral(self):
        """Derive dynamic confidence/margin thresholds from neutral noise."""
        neutral = self.profiles.get("neutral", {})
        vecs = neutral.get("vectors") or []
        if len(vecs) <= 10:
            # Keep defaults
            self.computed_thresholds = {"gate": 0.60, "margin": 0.08}
            return
        mat = np.array(vecs, dtype=float)
        # assume last column is neutral prob; drop it to inspect non-neutral noise
        if mat.shape[1] > 1:
            non_neutral = mat[:, :-1]
        else:
            non_neutral = mat
        max_noise = float(np.max(non_neutral))
        std_noise = float(np.std(non_neutral))
        suggested_gate = max_noise + 2.0 * std_noise
        gate = float(np.clip(suggested_gate, 0.45, 0.85))
        margin = float(max(0.05, std_noise * 1.5))
        self.computed_thresholds = {"gate": gate, "margin": margin}
        print(f"📊 Auto-Tuned Gate: {gate:.2f}, Margin: {margin:.2f}")

    def _warmup_session(self, cam, seconds: float = 2.0, show_preview: bool = True):
        """Read a few frames and run dummy DeepFace/FaceMesh to avoid cold-start drops."""
        print("Warm-up: initialisiere Kamera/Modelle ... (Vorschau aktiv)")
        start = time.time()
        deepface_ran = False
        while time.time() - start < seconds:
            ret, frame = cam.read()
            if not ret:
                continue
            if show_preview:
                cv2.putText(
                    frame,
                    "Warm-up (Preview)",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("Emotion Profiling", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to abort warm-up
                    break
            boxes = detect_faces(frame)
            if boxes:
                extract_facemesh_features(frame, boxes[0])  # prime FaceMesh
                if not deepface_ran:
                    processed = preprocess_face(frame, boxes[0])
                    if processed is not None:
                        try:
                            DeepFace.analyze(
                                processed,
                                actions=["emotion"],
                                enforce_detection=False,
                                detector_backend="mediapipe",
                            )
                            deepface_ran = True
                        except Exception:
                            pass
        print("Warm-up abgeschlossen. Starte Aufnahme ...")

    def _wait_for_start(self, cam, emotion: str, instruction: str) -> bool:
        """Show live preview and wait for ENTER/SPACE to start, ESC/Q to abort."""
        print(f"\nProfil '{emotion}': {instruction}")
        print("Drücke ENTER/SPACE zum Starten, ESC oder q zum Abbrechen.")
        while True:
            ret, frame = cam.read()
            if not ret:
                continue
            overlay = frame.copy()
            cv2.putText(
                overlay,
                f"Profil: {emotion} (ENTER/SPACE starten, ESC/q abbrechen)",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.imshow("Emotion Profiling", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in {13, 10, 32}:  # Enter/Return or Space
                return True
            if key in {27, ord("q"), ord("Q")}:
                return False

    def _prepare_classifier_dataset(self):
        # collect all embeddings and labels
        X = []
        y = []
        max_len = 0
        for emotion, payload in self.profiles.items():
            vecs = payload.get("vectors", [])
            if not vecs:
                continue
            max_len = max(max_len, len(vecs))
        if max_len == 0:
            return None

        feature_len = None
        for emotion, payload in self.profiles.items():
            vecs = payload.get("vectors", [])
            feats = payload.get("features", [])
            if not vecs or not feats or len(vecs) != len(feats):
                continue
            vecs_np = np.array(vecs)
            feats_np = np.array(feats, dtype=float)
            if len(vecs_np) < max_len:
                idx = self.rng.integers(0, len(vecs_np), size=max_len - len(vecs_np))
                vecs_np = np.concatenate([vecs_np, vecs_np[idx]], axis=0)
                feats_np = np.concatenate([feats_np, feats_np[idx]], axis=0)
            elif len(vecs_np) > max_len:
                idx = self.rng.choice(len(vecs_np), size=max_len, replace=False)
                vecs_np = vecs_np[idx]
                feats_np = feats_np[idx]
            feats_np = self._center_features(feats_np)
            combined = np.hstack([vecs_np, feats_np])
            feature_len = combined.shape[1]
            X.append(combined)
            y.extend([emotion] * combined.shape[0])
        if not X or feature_len is None:
            return None
        X_balanced = np.concatenate(X, axis=0)
        return X_balanced, np.array(y), feature_len

    def _init_classifier(self, model_name: str):
        name = (model_name or "logreg").lower()
        if name == "logreg":
            clf = LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=1000,
                n_jobs=None,
            )
        elif name == "svm":
            clf = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
        elif name == "random_forest":
            clf = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                class_weight="balanced",
                random_state=42,
            )
        elif name == "lightgbm":
            if LGBMClassifier is None:
                raise RuntimeError(
                    "LightGBM ist nicht installiert. Bitte `pip install lightgbm` ausführen oder TRAINED_MODEL ändern."
                )
            clf = LGBMClassifier(
                objective="multiclass",
                num_class=len(EMOTION_PROFILES),
                class_weight="balanced",
                learning_rate=0.05,
                n_estimators=300,
                random_state=42,
            )
        else:
            raise ValueError(f"Unbekanntes Modell: {model_name}")
        return name, clf

    def _build_classifier_payload(self, model, model_type, feature_len, scaler):
        classes_attr = getattr(model, "classes_", None)
        payload = {
            "model_type": model_type,
            "feature_length": feature_len,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "model_classes": list(classes_attr) if classes_attr is not None else [],
            "model_blob": self._serialize_model(model),
        }
        if model_type == "logreg":
            payload["classes"] = list(classes_attr) if classes_attr is not None else []
            payload["coef"] = getattr(model, "coef_", np.empty((0,))).tolist()
            payload["intercept"] = getattr(model, "intercept_", np.empty((0,))).tolist()
        return payload

    def _serialize_model(self, model) -> str:
        buffer = BytesIO()
        joblib.dump(model, buffer)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _save_snapshot(self):
        if not self.profiles:
            return
        data = {"profiles": {}}
        for emotion, payload in self.profiles.items():
            data["profiles"][emotion] = {
                "vectors": [vec.tolist() for vec in payload.get("vectors", [])],
                "features": [feat.tolist() for feat in payload.get("features", [])],
            }
        if self.neutral_feature_mean is not None:
            data["neutral_feature_mean"] = self.neutral_feature_mean.tolist()
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with self.snapshot_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"?? Emotion-Snapshot gespeichert unter: {self.snapshot_path}")

    def load_snapshot(self):
        if not self.snapshot_path.exists():
            print("?? Kein gespeicherter Emotion-Snapshot gefunden.")
            return False
        with self.snapshot_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        profiles = data.get("profiles")
        if not profiles:
            print("?? Snapshot leer oder ung?ltig.")
            return False
        self.profiles = {}
        for emotion, payload in profiles.items():
            vectors = [np.array(vec) for vec in payload.get("vectors", [])]
            features = [np.array(feat) for feat in payload.get("features", [])]
            self.profiles[emotion] = {
                "vectors": vectors,
                "features": features,
                "samples": [],
            }
        neutral_mean = data.get("neutral_feature_mean")
        if neutral_mean is not None:
            self.neutral_feature_mean = np.array(neutral_mean, dtype=float)
        else:
            self.neutral_feature_mean = self._compute_neutral_feature_mean()
        print(f"?? Emotion-Snapshot geladen ({self.snapshot_path})")
        return True

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rest-Face Kalibrierung für Moody.")
    parser.add_argument(
        "-e",
        "--emotion",
        action="append",
        help="Name einer Emotion (z. B. happy). Mehrfach nutzbar, um mehrere gezielt aufzunehmen.",
    )
    parser.add_argument(
        "--list-emotions",
        action="store_true",
        help="Zeigt alle verfügbaren Emotionen samt Anleitung.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=12,
        help="Aufnahmedauer pro Emotion (Sekunden). Standard: 12",
    )
    parser.add_argument(
        "--analyze-every",
        type=int,
        default=5,
        help="Frame-Schrittweite für die Analyse. Standard: 5",
    )
    args = parser.parse_args()

    if args.list_emotions:
        print("Verfügbare Emotionen:")
        for emotion, instruction in EMOTION_PROFILES:
            print(f" - {emotion}: {instruction}")
        raise SystemExit(0)

    calibrator = RestFaceCalibrator(model_path=REST_FACE_MODEL_PATH)
    if args.emotion:
        calibrator.load_snapshot()
    if calibrator.record_emotions(
        emotions=args.emotion,
        duration=args.duration,
        analyze_every=args.analyze_every,
    ):
        calibrator.train()
        calibrator.save_model()
        calibrator.visualize_space()
