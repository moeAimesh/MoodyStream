"""Offline feature-importance evaluation utilities.

Usage:
    python -m utils.eval_features [--mode ablation|permutation|coef]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from utils.settings import IMPORTANCE_TEST, REST_FACE_MODEL_PATH

GROUP_SPECS: List[Tuple[str, int]] = [
    ("deepface", 7),
    ("brow_raise", 7),
    ("brow_lower", 5),
    ("lid_aperture", 7),
    ("cheek_raise", 7),
    ("nose_flare", 5),
    ("mouth_corner", 8),
    ("mouth_depressor", 7),
]


def _feature_slices() -> List[Tuple[str, slice]]:
    slices: List[Tuple[str, slice]] = []
    cursor = 0
    for name, length in GROUP_SPECS:
        slices.append((name, slice(cursor, cursor + length)))
        cursor += length
    return slices


def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    profiles = data.get("profiles") or {}
    X_parts: List[np.ndarray] = []
    labels: List[str] = []
    for emotion, payload in profiles.items():
        vectors = np.array(payload.get("vectors", []), dtype=np.float32)
        features = np.array(payload.get("feature_vectors", []), dtype=np.float32)
        if vectors.size == 0 or features.size == 0:
            continue
        if vectors.shape[0] != features.shape[0]:
            count = min(vectors.shape[0], features.shape[0])
            vectors = vectors[:count]
            features = features[:count]
        combined = np.hstack([vectors, features])
        X_parts.append(combined)
        labels.extend([emotion] * combined.shape[0])
    if not X_parts:
        raise RuntimeError("Keine gespeicherten Samples gefunden – bitte Setup erneut ausführen.")
    X = np.concatenate(X_parts, axis=0)
    y = np.array(labels)
    return X, y


def build_model() -> LogisticRegression:
    return LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=3000,
    )


def stratified_splitter(y: Sequence[str]) -> StratifiedKFold:
    _, inv = np.unique(y, return_inverse=True)
    class_counts = np.bincount(inv)
    min_class = int(class_counts.min()) if class_counts.size else 2
    n_splits = max(2, min(5, min_class))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


def cross_val_score(X: np.ndarray, y: np.ndarray) -> float:
    splitter = stratified_splitter(y)
    scores: List[float] = []
    for train_idx, test_idx in splitter.split(X, y):
        clf = build_model()
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        scores.append(accuracy_score(y[test_idx], preds))
    return float(np.mean(scores))


def run_ablation(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    base_score = cross_val_score(X, y)
    results = {"baseline": base_score}
    for name, slc in _feature_slices():
        mask = np.ones(X.shape[1], dtype=bool)
        mask[slc] = False
        score = cross_val_score(X[:, mask], y)
        results[name] = score
    return results


def run_permutation(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    splitter = stratified_splitter(y)
    train_idx, test_idx = next(splitter.split(X, y))
    clf = build_model()
    clf.fit(X[train_idx], y[train_idx])
    result = permutation_importance(
        clf,
        X[test_idx],
        y[test_idx],
        n_repeats=20,
        random_state=42,
        n_jobs=1,
    )
    scores = {}
    for name, slc in _feature_slices():
        block = np.abs(result.importances[slc])
        scores[name] = float(block.mean())
    return scores


def run_coef(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    clf = build_model()
    clf.fit(X, y)
    coef = np.abs(clf.coef_)
    scores = {}
    for name, slc in _feature_slices():
        scores[name] = float(coef[:, slc].mean())
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AU feature importance.")
    parser.add_argument(
        "--mode",
        choices=["ablation", "permutation", "coef"],
        default=IMPORTANCE_TEST.lower(),
        help="Importance test type.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=REST_FACE_MODEL_PATH,
        help="Pfad zur gespeicherten rest_face_model.json",
    )
    args = parser.parse_args()
    X, y = load_dataset(args.model)
    if args.mode == "ablation":
        scores = run_ablation(X, y)
    elif args.mode == "permutation":
        scores = run_permutation(X, y)
    else:
        scores = run_coef(X, y)

    print(f"[importance] mode={args.mode}")
    for name, score in scores.items():
        print(f"  {name:15s} -> {score:.4f}")


if __name__ == "__main__":
    main()
