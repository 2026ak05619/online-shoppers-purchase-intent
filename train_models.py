"""
train_models.py
-------------------------------------------------------------------
Trains the five required classifiers on ONE dataset, evaluates them,
and saves:
    model/<name>.joblib        -> full pipeline (preprocessing + model)
    model/label_encoder.joblib -> target encoder (if target is text)
    model/metadata.joblib      -> feature list, class names, target name
    test_data.csv              -> hold-out test split (features + target)
    results/metrics_comparison.csv / .md -> comparison table for README

HOW TO USE
    1. Put your dataset CSV next to this file (e.g. data/raw_dataset.csv)
    2. Edit the CONFIG block below (DATA_PATH, TARGET_COLUMN, DROP_COLUMNS)
    3. Run:  python train_models.py
-------------------------------------------------------------------
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ===================== CONFIG — EDIT THIS BLOCK ONLY =====================
DATA_PATH = "data/raw_dataset.csv"   # path to your downloaded dataset
TARGET_COLUMN = "Revenue"             # EXACT name of the label column
DROP_COLUMNS = []                    # e.g. ["id", "customer_id"] — IDs leak nothing useful
CSV_SEPARATOR = ","                  # some UCI files use ";"
TEST_SIZE = 0.25
RANDOM_STATE = 42
MODEL_DIR = "model"
RESULTS_DIR = "results"
# =========================================================================


def load_data():
    df = pd.read_csv(DATA_PATH, sep=CSV_SEPARATOR)
    df.columns = [c.strip() for c in df.columns]
    if DROP_COLUMNS:
        df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    df = df.drop_duplicates().reset_index(drop=True)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target '{TARGET_COLUMN}' not found. Columns are: {list(df.columns)}"
        )
    df = df.dropna(subset=[TARGET_COLUMN])
    return df


def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))

    # sparse_threshold=0 keeps output dense so GaussianNB works
    return ColumnTransformer(transformers, sparse_threshold=0), numeric_cols, categorical_cols


def get_models():
    """Tune these hyper-parameters yourself — it is part of the marks and
    it is what makes your submission different from everyone else's."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=11, weights="distance"),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel="rbf", C=1.0, probability=True, random_state=RANDOM_STATE),
    }


def compute_metrics(y_true, y_pred, y_proba, n_classes):
    """Returns the six required metrics. Handles binary and multi-class."""
    if n_classes == 2:
        avg = "binary"
        auc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        avg = "macro"
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_data()
    print(f"Dataset shape after cleaning : {df.shape}")
    print(f"Features (excl. target)      : {df.shape[1] - 1}  (assignment minimum is 12)")
    print(f"Instances                    : {df.shape[0]}  (assignment minimum is 500)")

    X = df.drop(columns=[TARGET_COLUMN])
    y_raw = df[TARGET_COLUMN]

    # Encode the target to integers (keeps SVM/AUC happy, reversible for display)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw.astype(str))
    class_names = [str(c) for c in label_encoder.classes_]
    n_classes = len(class_names)
    print(f"Target '{TARGET_COLUMN}' classes : {class_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Save the hold-out test split as the CSV students upload to Streamlit
    test_out = X_test.copy()
    test_out[TARGET_COLUMN] = label_encoder.inverse_transform(y_test)
    test_out.to_csv("test_data.csv", index=False)
    print(f"Saved test_data.csv          : {test_out.shape[0]} rows")

    preprocessor, num_cols, cat_cols = build_preprocessor(X)
    print(f"Numeric columns: {len(num_cols)} | Categorical columns: {len(cat_cols)}")

    rows = []
    for name, estimator in get_models().items():
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("classifier", estimator),
        ])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)
        scores = compute_metrics(y_test, y_pred, y_proba, n_classes)
        scores_row = {"ML Model Name": name, **{k: round(v, 4) for k, v in scores.items()}}
        rows.append(scores_row)

        filename = os.path.join(MODEL_DIR, name.lower().replace(" ", "_") + ".joblib")
        joblib.dump(pipe, filename)
        print(f"  trained + saved -> {filename}  (accuracy {scores['Accuracy']:.4f})")

    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.joblib"))
    joblib.dump(
        {
            "feature_columns": list(X.columns),
            "target_column": TARGET_COLUMN,
            "class_names": class_names,
            "n_classes": n_classes,
            "sklearn_version": sklearn.__version__,
        },
        os.path.join(MODEL_DIR, "metadata.joblib"),
    )

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(RESULTS_DIR, "metrics_comparison.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "metrics_comparison.md"), "w") as f:
        f.write(results.to_markdown(index=False))

    print("\n================ COMPARISON TABLE (paste into README) ================")
    print(results.to_markdown(index=False))
    print("\nBest by F1 :", results.loc[results["F1"].idxmax(), "ML Model Name"])
    print("Best by MCC:", results.loc[results["MCC"].idxmax(), "ML Model Name"])
    print(f"\nscikit-learn version used: {sklearn.__version__}")
    print("Pin this EXACT version in requirements.txt or Streamlit Cloud may fail to load the models.")


if __name__ == "__main__":
    main()
