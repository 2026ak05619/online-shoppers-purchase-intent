"""
app.py — Streamlit front-end for the five trained classifiers.

It loads the saved pipelines from model/, scores an uploaded test CSV,
and shows metrics, a confusion matrix and a classification report.
No retraining happens here.
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------- settings
# TODO: change the title / caption / dataset blurb to match YOUR dataset.
APP_TITLE = "Purchase Intent Predictor"
APP_CAPTION = "Scoring five classifiers on e-commerce session data from the UCI Online Shoppers dataset."
MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_artifacts():
    models, missing = {}, []
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            missing.append(path)
    encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    metadata = joblib.load(os.path.join(MODEL_DIR, "metadata.joblib"))
    return models, encoder, metadata, missing


def evaluate(pipeline, X, y_true, n_classes):
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)
    if n_classes == 2:
        avg = "binary"
        auc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        avg = "macro"
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    scores = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return scores, y_pred


# ---------------------------------------------------------------- layout
models, encoder, metadata, missing = load_artifacts()
target_col = metadata["target_column"]
feature_cols = metadata["feature_columns"]
class_names = metadata["class_names"]
n_classes = metadata["n_classes"]

st.title(APP_TITLE)
st.caption(APP_CAPTION)
st.info("The target is 84.37% negative, so always predicting no-purchase already scores 0.8437 accuracy. Judge these models on MCC and recall, not accuracy.")

if missing:
    st.error("These model files are missing from the repository: " + ", ".join(missing))
    st.stop()

with st.sidebar:
    st.header("Controls")
    chosen_model = st.selectbox("Model", list(models.keys()))
    st.divider()
    st.subheader("Expected CSV")
    st.write(f"Target column: `{target_col}`")
    st.write(f"Feature columns: {len(feature_cols)}")
    with st.expander("Show required columns"):
        st.code("\n".join(feature_cols))
    st.caption(f"Classes: {', '.join(class_names)}")

uploaded = st.file_uploader("Test data (CSV with features + target column)", type=["csv"])

if uploaded is None:
    st.info("Upload `test_data.csv` from the repository to score the models.")
    st.stop()

df = pd.read_csv(uploaded)
st.success(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
with st.expander("Preview uploaded data"):
    st.dataframe(df.head(10), use_container_width=True)

# ---------------------------------------------------------------- validation
if target_col not in df.columns:
    st.error(f"The CSV has no `{target_col}` column, so metrics cannot be computed.")
    st.stop()

absent = [c for c in feature_cols if c not in df.columns]
if absent:
    st.error("Missing feature columns: " + ", ".join(absent))
    st.stop()

X = df[feature_cols]
try:
    y_true = encoder.transform(df[target_col].astype(str))
except ValueError:
    st.error(f"`{target_col}` contains labels the models were never trained on.")
    st.stop()

# ---------------------------------------------------------------- results
tab_single, tab_all = st.tabs(["Selected model", "All five models"])

with tab_single:
    scores, y_pred = evaluate(models[chosen_model], X, y_true, n_classes)
    st.subheader(chosen_model)

    cols = st.columns(6)
    for col, (metric, value) in zip(cols, scores.items()):
        col.metric(metric, f"{value:.4f}")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="rocket_r", cbar=False,
            xticklabels=class_names, yticklabels=class_names, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.markdown("**Classification report**")
        report = classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

    out = df.copy()
    out["predicted_" + target_col] = encoder.inverse_transform(y_pred)
    st.download_button(
        "Download predictions",
        out.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{chosen_model.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

with tab_all:
    st.subheader("Comparison on the uploaded test data")
    rows = []
    for name, pipe in models.items():
        s, _ = evaluate(pipe, X, y_true, n_classes)
        rows.append({"ML Model Name": name, **s})
    table = pd.DataFrame(rows).set_index("ML Model Name").round(4)
    st.dataframe(table.style.highlight_max(axis=0, color="#c6efce"), use_container_width=True)

    metric_to_plot = st.selectbox("Chart a metric", list(table.columns), index=0)
    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    ax2.bar(table.index, table[metric_to_plot], color="#B5651D")
    ax2.set_ylabel(metric_to_plot)
    ax2.set_ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    st.pyplot(fig2)

    best = table["F1"].idxmax()
    st.info(f"Highest F1 on this upload: **{best}** ({table.loc[best, 'F1']:.4f})")
