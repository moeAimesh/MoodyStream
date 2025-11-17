#!/usr/bin/env python
"""Visualize pairwise emotion distances via heatmap and dendrogram."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from utils.settings import REST_FACE_MODEL_PATH


def _load_class_means(model_path: Path) -> Tuple[List[str], np.ndarray, np.ndarray]:
    data = json.loads(model_path.read_text(encoding="utf-8"))
    profiles = data.get("profiles")
    if not profiles:
        raise RuntimeError("Keine Profile im Modell gefunden. Bitte Setup erneut ausführen.")

    labels: List[str] = []
    means: List[np.ndarray] = []
    stacked_samples: List[np.ndarray] = []

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
        labels.append(emotion)
        means.append(combined.mean(axis=0))
        stacked_samples.append(combined)

    if not labels:
        raise RuntimeError("Keine gültigen Samples gefunden.")
    return labels, np.vstack(means), np.vstack(stacked_samples)


def _compute_distance_matrix(means: np.ndarray, samples: np.ndarray, metric: str) -> np.ndarray:
    if metric == "mahalanobis":
        cov = np.cov(samples, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6
        inv_cov = np.linalg.pinv(cov)
        diff = means[:, None, :] - means[None, :, :]
        dist = np.sqrt(np.einsum("...i,ij,...j->...", diff, inv_cov, diff))
    else:
        diff = means[:, None, :] - means[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
    return dist


def _plot_heatmap(dist_matrix: np.ndarray, labels: List[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(dist_matrix, cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Pairwise Emotion Distances")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Distance")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{dist_matrix[i, j]:.2f}", ha="center", va="center", color="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_dendrogram(dist_matrix: np.ndarray, labels: List[str], path: Path, method: str) -> None:
    condensed = squareform(dist_matrix, checks=False)
    linkage_matrix = linkage(condensed, method=method)
    fig, ax = plt.subplots(figsize=(8, 6))
    dendrogram(linkage_matrix, labels=labels, ax=ax, orientation="right")
    ax.set_title(f"Hierarchical Clustering ({method})")
    ax.set_xlabel("Distance")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze pairwise emotion cluster distances.")
    parser.add_argument("--model", type=Path, default=REST_FACE_MODEL_PATH, help="Path to rest_face_model.json")
    parser.add_argument(
        "--metric",
        choices=["euclidean", "mahalanobis"],
        default="euclidean",
        help="Distance metric used for pairwise comparison.",
    )
    parser.add_argument(
        "--linkage",
        choices=["average", "complete", "single"],
        default="average",
        help="Linkage method for dendrogram.",
    )
    parser.add_argument(
        "--heatmap-output",
        type=Path,
        default=Path("plots/cluster_distance_heatmap.png"),
        help="File to store the heatmap.",
    )
    parser.add_argument(
        "--dendrogram-output",
        type=Path,
        default=Path("plots/cluster_dendrogram.png"),
        help="File to store the dendrogram.",
    )
    args = parser.parse_args()

    labels, means, samples = _load_class_means(args.model)
    dist_matrix = _compute_distance_matrix(means, samples, metric=args.metric)
    _plot_heatmap(dist_matrix, labels, args.heatmap_output)
    _plot_dendrogram(dist_matrix, labels, args.dendrogram_output, method=args.linkage)
    print(f"✅ Heatmap gespeichert unter {args.heatmap_output}")
    print(f"✅ Dendrogram gespeichert unter {args.dendrogram_output}")


if __name__ == "__main__":
    main()
