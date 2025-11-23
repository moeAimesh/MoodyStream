#!/usr/bin/env python
"""Visualize emotion decision regions via PCA projection."""

from __future__ import annotations

import argparse
import base64
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

from utils.settings import (
    CLUSTER_NEIGHBOR_RATIO,
    CLUSTER_RADIUS_QUANTILE,
    REST_FACE_MODEL_PATH,
)

try:
    from umap import UMAP
except ImportError:  # pragma: no cover - optional dependency
    UMAP = None


def load_dataset(model_path: Path):
    data = json.loads(model_path.read_text(encoding="utf-8"))
    profiles = data.get("profiles")
    if not profiles:
        raise RuntimeError("Keine Profile im Modell gefunden. Bitte Setup erneut ausführen.")

    samples = []
    labels = []
    grouped: Dict[str, List[np.ndarray]] = {}
    for emotion, payload in profiles.items():
        vectors = payload.get("vectors") or []
        features = payload.get("feature_vectors") or []
        if not vectors or not features:
            continue
        count = min(len(vectors), len(features))
        if count == 0:
            continue
        vecs = np.asarray(vectors[:count], dtype=float)
        feats = np.asarray(features[:count], dtype=float)
        combined = np.hstack([vecs, feats])
        samples.append(combined)
        labels.extend([emotion] * combined.shape[0])
        grouped.setdefault(emotion, []).append(combined)
    if not samples:
        raise RuntimeError("Keine Samples verfügbar – Snapshot leer?")
    grouped = {label: np.vstack(chunks) for label, chunks in grouped.items()}
    return np.concatenate(samples, axis=0), np.array(labels), data.get("classifier"), grouped


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


def scale_vector(vector: np.ndarray, classifier_meta: dict) -> np.ndarray:
    if not classifier_meta:
        return vector
    mean = classifier_meta.get("scaler_mean")
    scale = classifier_meta.get("scaler_scale")
    if mean is None or scale is None:
        return vector
    mean = np.array(mean)
    scale = np.array(scale)
    denom = np.where(scale == 0, 1.0, scale)
    return (vector - mean) / denom


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


def compute_umap_embedding(X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, Optional[object]]:
    """Return a low-dimensional embedding using UMAP or fall back to t-SNE."""
    if UMAP is not None:
        reducer = UMAP(n_components=n_components, random_state=42)
        return reducer.fit_transform(X), reducer
    print("[!] Paket 'umap-learn' nicht gefunden - falle auf t-SNE zurueck.")
    tsne = TSNE(n_components=n_components, random_state=42, init="pca", learning_rate="auto")
    return tsne.fit_transform(X), None


def _compute_cluster_limits(grouped: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
    labels = list(grouped.keys())
    if not labels:
        return {}
    means = {label: grouped[label].mean(axis=0) for label in labels}
    mean_matrix = np.vstack([means[label] for label in labels])
    dist_matrix = np.linalg.norm(mean_matrix[:, None, :] - mean_matrix[None, :, :], axis=2)
    limits: Dict[str, Dict[str, np.ndarray]] = {}
    for idx, label in enumerate(labels):
        samples = grouped[label]
        if samples.size == 0:
            continue
        dists = np.linalg.norm(samples - means[label], axis=1)
        if dists.size == 0:
            continue
        q_val = float(np.quantile(dists, CLUSTER_RADIUS_QUANTILE))
        neighbor = np.delete(dist_matrix[idx], idx)
        neighbor_limit = float(np.min(neighbor)) * CLUSTER_NEIGHBOR_RATIO if neighbor.size else q_val
        radius = max(1e-6, min(neighbor_limit, q_val))
        limits[label] = {
            "radius": radius,
            "quantile": q_val,
            "mean": means[label],
        }
    return limits


def _plot_boundaries(
    ax,
    embedding: np.ndarray,
    labels: np.ndarray,
    cluster_limits: Dict[str, Dict[str, np.ndarray]],
    grouped: Dict[str, np.ndarray],
    color_map: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    summary = {}
    if not cluster_limits:
        return summary
    for label, params in cluster_limits.items():
        mask = labels == label
        if not mask.any():
            continue
        coords = embedding[mask]
        center = coords.mean(axis=0)
        emb_dists = np.linalg.norm(coords - center, axis=1)
        q_emb = float(np.quantile(emb_dists, CLUSTER_RADIUS_QUANTILE)) if emb_dists.size else 0.0
        q_high = max(params.get("quantile") or 1e-6, 1e-6)
        scale = q_emb / q_high if q_emb > 0 else 1.0
        radius_emb = params["radius"] * scale
        summary[label] = {
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            "radius_emb": float(radius_emb),
        }
        circ = plt.Circle(
            center,
            radius_emb,
            color=color_map.get(label, "black"),
            fill=False,
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
        )
        ax.add_patch(circ)
    return summary


def _analyze_sample_vector(vector: np.ndarray, cluster_limits: Dict[str, Dict[str, np.ndarray]]):
    best_label = None
    best_dist = None
    best_radius = None
    for label, params in cluster_limits.items():
        mean_vec = params.get("mean")
        if mean_vec is None:
            continue
        dist = float(np.linalg.norm(vector - mean_vec))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_label = label
            best_radius = params.get("radius")
    if best_label is None:
        return None, None, None, False
    allowed = best_radius is None or best_dist <= best_radius
    return best_label, best_dist, best_radius, allowed


def _load_sample_from_image(image_path: Path) -> np.ndarray:
    import cv2
    from deepface import DeepFace

    from detection.face_detection import detect_faces
    from detection.facemesh_features import extract_facemesh_features
    from detection.preprocessing import preprocess_face

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Konnte Bild nicht laden: {image_path}")
    boxes = detect_faces(frame)
    if not boxes:
        raise RuntimeError("Kein Gesicht im Bild gefunden.")
    bbox = boxes[0]
    processed = preprocess_face(frame, bbox)
    if processed is None:
        raise RuntimeError("Gesicht konnte nicht vorverarbeitet werden.")
    features = extract_facemesh_features(frame, bbox)
    if features is None:
        raise RuntimeError("Facemesh-Features konnten nicht extrahiert werden.")
    result = DeepFace.analyze(
        processed,
        actions=["emotion"],
        enforce_detection=False,
        detector_backend="mediapipe",
    )
    emotion_vec = np.array(list(result[0]["emotion"].values()), dtype=float)
    return np.concatenate([emotion_vec, features])


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
    parser.add_argument(
        "--show-boundaries",
        action="store_true",
        help="Zeigt berechnete Cluster-Radien (nur bei UMAP-Plot).",
    )
    parser.add_argument(
        "--sample-image",
        type=Path,
        default=None,
        help="Optionales Bild, dessen Punkt auf der UMAP-Projektion angezeigt wird.",
    )
    args = parser.parse_args()

    X, y, classifier_meta, grouped = load_dataset(args.model)
    classifier_meta = classifier_meta or {}
    cluster_limits = _compute_cluster_limits(grouped)
    sample_vector = None
    if args.sample_image is not None:
        try:
            sample_vector = _load_sample_from_image(args.sample_image)
        except Exception as exc:
            print(f"⚠️ Zusatz-Bild konnte nicht verarbeitet werden: {exc}")
            sample_vector = None
    X_scaled = apply_scaler(X, classifier_meta)
    sample_vector_scaled = scale_vector(sample_vector, classifier_meta) if sample_vector is not None else None

    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    y_encoded = label_encoder.transform(y)

    classifier_model, _ = load_classifier_model(classifier_meta)

    plot_type = args.plot_type.lower()

    colors = plt.cm.get_cmap("tab10", len(label_encoder.classes_))
    color_map = {label: colors(idx) for idx, label in enumerate(label_encoder.classes_)}

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
            plt.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                label=emotion,
                s=25,
                alpha=0.85,
                color=color_map.get(emotion),
            )
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
                color=color_map.get(emotion),
            )
        ax.set_title("PCA Projection (3D)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.legend(loc="best")

    elif plot_type == "umap":
        embedding, reducer = compute_umap_embedding(X_scaled, n_components=2)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        for emotion in label_encoder.classes_:
            mask = y == emotion
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                label=emotion,
                s=35,
                alpha=0.85,
                color=color_map.get(emotion),
            )
        boundary_summary = {}
        if args.show_boundaries:
            boundary_summary = _plot_boundaries(ax, embedding, y, cluster_limits, grouped, color_map)
        if sample_vector_scaled is not None:
            if reducer is None:
                print("⚠️ Zusatz-Bild kann nur mit UMAP dargestellt werden (kein transform für t-SNE).")
            else:
                sample_point = reducer.transform([sample_vector_scaled])[0]
                best_label, dist, radius, allowed = _analyze_sample_vector(sample_vector, cluster_limits)
                marker_color = "black" if allowed else "red"
                ax.scatter(
                    sample_point[0],
                    sample_point[1],
                    marker="*",
                    s=180,
                    edgecolors="white",
                    linewidths=1.5,
                    color=marker_color,
                    label="sample" if "sample" not in ax.get_legend_handles_labels()[1] else None,
                )
                print(
                    f"🔎 Zusatz-Bild -> nächster Cluster '{best_label}' "
                    f"(Distanz {dist:.2f}, Grenze {radius:.2f if radius else float('nan')}) "
                    f"=> {'innerhalb' if allowed else 'außerhalb'}"
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

#Z:\CODING\UNI\MOODY\
#
