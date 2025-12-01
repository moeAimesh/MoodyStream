import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

def main():
    data = np.load(r"C:\GitHub\MoodyStream\detection\detectors\gestures\gesture_dataset.npz", allow_pickle=True)
    X = data["X"]  # Form: (N, 63)
    y = data["y"]  # Form: (N,)

    print("X-Shape:", X.shape)
    print("y-Shape:", y.shape)

    # Train/Test-Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline: StandardScaler + SVM
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True))
    ])

    print("Trainiere SVM...")
    clf.fit(X_train, y_train)

    # Evaluation
    y_pred = clf.predict(X_test)
    print("Ergebnisse auf Test-Set:")
    print(classification_report(y_test, y_pred))

    # Modell speichern
    joblib.dump(clf, "svm_gesture_models.pkl")

if __name__ == "__main__":
    main()
