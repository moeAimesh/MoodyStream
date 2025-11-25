#!/usr/bin/env python
"""Utility script to drop outlier samples from the emotion snapshot and retrain the model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from setup.face_setup import RestFaceCalibrator
from utils.settings import REST_FACE_MODEL_PATH


def _stack_safe(items: List[np.ndarray]) -> np.ndarray:
    """Convert a list of (possibly list-based) vectors to a 2D numpy array."""
    return np.stack([np.asarray(vec, dtype=float) for vec in items])


def _combined_vectors(vectors: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Create a combined representation matching training inputs."""
    return np.hstack([vectors, features])


def _detect_mask(
    data: np.ndarray,
    method: str,
    contamination: float,
    radius_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "lof":
        n_neighbors = min(20, data.shape[0] - 1)
        if n_neighbors < 2:
            # Not enough samples to run LOF – keep everything.
            return np.ones(data.shape[0], dtype=bool), np.ones(data.shape[0])
        detector = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=min(contamination, 0.5),
        )
        labels = detector.fit_predict(data)
        scores = detector.negative_outlier_factor_
        mask = labels == 1
        return mask, scores

    if method == "radius":
        mean_vec = np.mean(data, axis=0)
        distances = np.linalg.norm(data - mean_vec, axis=1)
        dist_mean = float(np.mean(distances))
        dist_std = float(np.std(distances) or 1e-6)
        threshold = dist_mean + radius_sigma * dist_std
        mask = distances <= threshold
        return mask, distances

    detector = IsolationForest(
        contamination=min(contamination, 0.5),
        n_estimators=300,
        random_state=42,
    )
    labels = detector.fit_predict(data)
    scores = detector.score_samples(data)
    mask = labels == 1
    return mask, scores


def filter_outliers(
    calibrator: RestFaceCalibrator,
    method: str,
    contamination: float,
    radius_sigma: float,
) -> Dict[str, Dict[str, float]]:
    """Filter vectors/features using an unsupervised detector on the combined vector."""
    summary: Dict[str, Dict[str, float]] = {}
    for emotion, payload in calibrator.profiles.items():
        vectors = payload.get("vectors") or []
        features = payload.get("features") or []
        count = min(len(vectors), len(features))
        if count < 2:
            summary[emotion] = {"original": count, "kept": count, "removed": 0, "score_min": float("nan")}
            continue

        vectors_np = _stack_safe(vectors[:count])
        features_np = _stack_safe(features[:count])
        combined = _combined_vectors(vectors_np, features_np)

        mask, scores = _detect_mask(
            combined,
            method=method,
            contamination=contamination,
            radius_sigma=radius_sigma,
        )
        kept = int(mask.sum())
        removed = int(count - kept)

        if removed:
            payload["vectors"] = [np.array(vec) for vec in vectors_np[mask]]
            payload["features"] = [np.array(feat) for feat in features_np[mask]]
        summary[emotion] = {
            "original": count,
            "kept": kept,
            "removed": removed,
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter outliers from the saved emotion snapshot.")
    parser.add_argument(
        "--model",
        type=Path,
        default=REST_FACE_MODEL_PATH,
        help="Path to rest_face_model.json (used for saving results).",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Optional path to the snapshot file. Defaults to <model>.profiles.snapshot.json.",
    )
    parser.add_argument(
        "--method",
        choices=["lof", "iforest", "radius"],
        default="lof",
        help="Outlier detector to use: LocalOutlierFactor or IsolationForest.",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Estimated fraction of outliers per class (used by the detectors).",
    )
    parser.add_argument(
        "--radius-sigma",
        type=float,
        default=2.5,
        help="Only used for method=radius: keep samples within mean + sigma*std.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show the filtering summary without training/saving.",
    )
    args = parser.parse_args()

    calibrator = RestFaceCalibrator(model_path=args.model, snapshot_path=args.snapshot)
    if not calibrator.load_snapshot():
        raise SystemExit("Snapshot konnte nicht geladen werden.")

    summary = filter_outliers(
        calibrator,
        method=args.method,
        contamination=args.contamination,
        radius_sigma=args.radius_sigma,
    )
    removed_total = sum(item["removed"] for item in summary.values())

    print("=== Filter summary ===")
    for emotion, stats in summary.items():
        extra = (
            f"score_range=({stats['score_min']:.3f}, {stats['score_max']:.3f})"
            if not np.isnan(stats["score_min"])
            else "no-score"
        )
        print(
            f"{emotion:>8s}: kept {stats['kept']:3d}/{stats['original']:3d} "
            f"(removed {stats['removed']:2d}) {extra}"
        )

    if args.dry_run:
        return

    if removed_total == 0:
        print("Keine Ausreißer entfernt – trainiere/speichere trotzdem mit allen Samples.")

    calibrator.train()
    calibrator.save_model()
    print(f"✅ Modell aktualisiert und nach {args.model} geschrieben.")


if __name__ == "__main__":
    main()
