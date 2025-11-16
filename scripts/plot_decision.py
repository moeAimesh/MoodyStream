#!/usr/bin/env python
"""Visualize emotion decision regions via PCA projection."""

from __future__ import annotations

import argparse
import base64
import json
import os
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  needed for 3D plots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.settings import REST_FACE_MODEL_PATH

try:
    import umap
except ImportError:  # pragma: no cover - optional dependency
    umap = None


def load_dataset(model_path: Path):
    data = json.loads(model_path.read_text(encoding="utf-8"))
    profiles = data.get("profiles")
    if not profiles:
        raise RuntimeError("Keine Profile im Modell gefunden. Bitte Setup erneut ausführen.")

    samples = []
    labels = []
    for emotion, payload in profiles.items():
        vectors = payload.get("vectors") or []
        features = payload.get("feature_vectors") or []
        for vec, feat in zip(vectors, features):
            samples.append(np.concatenate([np.array(vec), np.array(feat)]))
            labels.append(emotion)
    if not samples:
        raise RuntimeError("Keine Samples verfügbar – Snapshot leer?")
    return np.array(samples), np.array(labels), data.get("classifier")


def apply_scaler(X: np.ndarray, classifier_meta: dict) -> np.ndarray:
    if not classifier_meta:
        return X
    mean = classifier_meta.get("scaler_mean")
    scale = classifier_meta.get("scaler_scale")
    if mean is None or scale is None:
        return X
    mean = np.array(mean)
    scale = np.array(scale)
    scale_safe = np.where(scale == 0, 1.0, scale)
    return (X - mean) / scale_safe


class LinearClassifier:
    """Simple multinomial logistic regression wrapper when only coefficients are stored."""

    def __init__(self, coef: np.ndarray, intercept: np.ndarray, classes: list[str]):
        self.coef_ = np.asarray(coef, dtype=float)
        self.intercept_ = np.asarray(intercept, dtype=float)
        self.classes_ = np.asarray(classes)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.coef_.T + self.intercept_
        logits -= np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


def load_classifier_model(classifier_meta: dict):
    """Load the serialized classifier (joblib blob or linear fallback) together with its classes."""
    if not classifier_meta:
        return None, []
    model_blob = classifier_meta.get("model_blob")
    classes = classifier_meta.get("model_classes") or []
    if model_blob:
        try:
            buffer = BytesIO(base64.b64decode(model_blob))
            model = joblib.load(buffer)
        except Exception as exc:  # pragma: no cover - best effort diagnostics
            raise RuntimeError(f"Konnte Klassifikator nicht laden: {exc}") from exc
        classes_attr = getattr(model, "classes_", None)
        if classes_attr is not None:
            classes = list(classes_attr)
        return model, classes
    coef = classifier_meta.get("coef")
    intercept = classifier_meta.get("intercept")
    class_labels = classifier_meta.get("classes") or classes
    if coef is not None and intercept is not None and class_labels:
        linear = LinearClassifier(np.array(coef), np.array(intercept), class_labels)
        return linear, list(class_labels)
    return None, classes


def predict_labels(model, X: np.ndarray) -> np.ndarray:
    """Predict labels from any classifier and return them as strings."""
    raw = model.predict(X)
    return np.asarray([str(label) for label in raw])


def compute_umap_embedding(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Return a low-dimensional embedding using UMAP or fall back to t-SNE."""
    if umap is not None:
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        return reducer.fit_transform(X)
    print("[!] Paket 'umap-learn' nicht gefunden - falle auf t-SNE zurueck.")
    tsne = TSNE(n_components=n_components, random_state=42, init="pca", learning_rate="auto")
    return tsne.fit_transform(X)


def main():
    parser = argparse.ArgumentParser(description="Plot emotion embeddings with multiple projections.")
    parser.add_argument(
        "--model",
        type=Path,
        default=REST_FACE_MODEL_PATH,
        help="Path to rest_face_model.json",
    )
    parser.add_argument("--output", type=Path, default=Path("plots/decision_regions.png"))
    default_plot_type = os.environ.get("PLOT_TYPE", "2d").lower()
    parser.add_argument(
        "--plot-type",
        choices=["2d", "3d", "umap"],
        default=default_plot_type,
        help="Art der Visualisierung: 2d (mit Entscheidungsflachen), 3d oder umap/tsne.",
    )
    args = parser.parse_args()

    X, y, classifier_meta = load_dataset(args.model)
    classifier_meta = classifier_meta or {}
    X_scaled = apply_scaler(X, classifier_meta)

    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    y_encoded = label_encoder.transform(y)

    classifier_model, _ = load_classifier_model(classifier_meta)

    plot_type = args.plot_type.lower()

    if plot_type == "2d":
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 400),
            np.linspace(y_min, y_max, 400),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]

        if classifier_model is not None:
            grid_scaled = pca.inverse_transform(grid)
            grid_labels = predict_labels(classifier_model, grid_scaled)
            Z = label_encoder.transform(grid_labels).reshape(xx.shape)
        else:
            print("[!] Kein gespeicherter Klassifikator gefunden - nutze temporaren SVC fuer die Visualisierung.")
            svc = SVC(kernel="rbf", gamma="scale", probability=True)
            svc.fit(X_pca, y_encoded)
            Z = svc.predict(grid).reshape(xx.shape)

        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.25, levels=len(label_encoder.classes_))
        for emotion in label_encoder.classes_:
            mask = y == emotion
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=emotion, s=25, alpha=0.85)
        plt.legend()
        plt.title("PCA Projection + Decision Regions (2D)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")

    elif plot_type == "3d":
        pca = PCA(n_components=3, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        for emotion in label_encoder.classes_:
            mask = y == emotion
            ax.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                X_pca[mask, 2],
                label=emotion,
                s=35,
                alpha=0.85,
            )
        ax.set_title("PCA Projection (3D)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.legend(loc="best")

    elif plot_type == "umap":
        embedding = compute_umap_embedding(X_scaled, n_components=2)
        plt.figure(figsize=(10, 8))
        for emotion in label_encoder.classes_:
            mask = y == emotion
            plt.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                label=emotion,
                s=35,
                alpha=0.85,
            )
        plt.legend()
        plt.title("UMAP/T-SNE Projection")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
    else:
        raise ValueError(f"Unbekannter plot_type: {plot_type}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
