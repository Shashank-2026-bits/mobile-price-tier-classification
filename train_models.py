"""Train and evaluate five classifiers on the Mobile Price Classification dataset."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "mobile_price.csv"
MODEL_DIR = ROOT / "model"
TEST_FILE = ROOT / "test_data.csv"

TARGET = "price_range"
CLASS_NAMES = ["Low cost", "Medium cost", "High cost", "Very high cost"]
SEED = 42

# Each entry: (file stem, pipeline, hyperparameter grid searched with 5-fold CV)
MODEL_SPECS = {
    "Logistic Regression": (
        "logistic_regression",
        Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))]),
        {"clf__C": [0.1, 1, 10, 100]},
    ),
    "Decision Tree": (
        "decision_tree",
        Pipeline([("clf", DecisionTreeClassifier(random_state=SEED))]),
        {"clf__max_depth": [4, 6, 8, 10, None], "clf__min_samples_leaf": [1, 5, 10]},
    ),
    "KNN": (
        "knn",
        Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
        {"clf__n_neighbors": [5, 9, 15, 21, 31], "clf__weights": ["uniform", "distance"]},
    ),
    "Naive Bayes": (
        "naive_bayes",
        Pipeline([("clf", GaussianNB())]),
        {"clf__var_smoothing": [1e-9, 1e-7, 1e-5]},
    ),
    "Support Vector Machine": (
        "svm",
        Pipeline([
            ("scaler", StandardScaler()),
            # Calibration supplies predict_proba, needed for AUC
            ("clf", CalibratedClassifierCV(SVC(random_state=SEED), ensemble=False)),
        ]),
        {"clf__estimator__kernel": ["linear", "rbf"], "clf__estimator__C": [0.1, 1, 10]},
    ),
}


def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_proba, multi_class="ovr", average="macro"),
        "Precision": precision_score(y, y_pred, average="macro"),
        "Recall": recall_score(y, y_pred, average="macro"),
        "F1": f1_score(y, y_pred, average="macro"),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def main():
    df = pd.read_csv(DATA_FILE)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    print(f"Dataset: {df.shape[0]} rows, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_test.assign(**{TARGET: y_test}).to_csv(TEST_FILE, index=False)
    print(f"Train: {len(X_train)} | Test: {len(X_test)} -> saved {TEST_FILE.name}")

    MODEL_DIR.mkdir(exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, best_params = [], {}

    for name, (stem, pipeline, grid) in MODEL_SPECS.items():
        search = GridSearchCV(pipeline, grid, cv=cv, scoring="f1_macro")
        search.fit(X_train, y_train)
        best = search.best_estimator_
        joblib.dump(best, MODEL_DIR / f"{stem}.pkl")

        scores = evaluate(best, X_test, y_test)
        rows.append({"ML Model Name": name, **scores})
        best_params[name] = {k.split("__")[-1]: v for k, v in search.best_params_.items()}
        print(f"{name:<24} CV F1={search.best_score_:.4f}  params={best_params[name]}")

    summary = pd.DataFrame(rows).set_index("ML Model Name").round(4)
    summary.to_csv(MODEL_DIR / "metrics_summary.csv")

    info = {
        "dataset": "Mobile Price Classification (Kaggle: iabhishekofficial/mobile-price-classification)",
        "target": TARGET,
        "features": list(X.columns),
        "class_names": CLASS_NAMES,
        "models": {name: f"{stem}.pkl" for name, (stem, _, _) in MODEL_SPECS.items()},
        "best_params": best_params,
        "sklearn_version": sklearn.__version__,
        "random_state": SEED,
    }
    (MODEL_DIR / "model_info.json").write_text(json.dumps(info, indent=2, default=str))

    print("\nTest-set metrics:")
    print(summary.to_string())


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
