# Predicting Online Purchase Intent from E-commerce Session Behaviour

M.Tech (AIML) — Machine Learning — Assignment 2
Name: Krishnendu Sinha | BITS ID: 2026AK05619

---

## a. Problem statement

Online retailers see far more browsing sessions than purchases. The goal here is to
predict, from 17 behavioural and technical attributes captured during a single visit,
whether that session will end in a transaction. A reliable prediction lets a site
trigger a retention offer while the visitor is still on the page rather than after
they have left. This is a binary classification problem on a strongly imbalanced
target: 84.37% of sessions do not convert, which makes accuracy a poor measure
of success and makes MCC and recall the metrics that matter.

## b. Dataset description

| Item | Value |
|---|---|
| Source | UCI Machine Learning Repository — Online Shoppers Purchasing Intention Dataset |
| URL | https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset |
| Instances (after removing duplicates) | 12,205 |
| Features used | 17 (14 numeric, 3 categorical) |
| Target column | `Revenue` |
| Classes | `False` (no purchase), `True` (purchase) |
| Class balance | `False` 84.37% (10,297) / `True` 15.63% (1,908) |
| Missing values | None in the source file; the pipeline still applies median / most-frequent imputation as a safeguard |
| Train/test split | 75% / 25%, stratified on `Revenue`, `random_state=42` (9,153 train / 3,052 test) |

**Feature list**

| # | Feature | Type | Meaning |
|---|---|---|---|
| 1 | Administrative | numeric | Number of account/admin pages viewed |
| 2 | Administrative_Duration | numeric | Seconds spent on those pages |
| 3 | Informational | numeric | Number of informational pages viewed |
| 4 | Informational_Duration | numeric | Seconds spent on those pages |
| 5 | ProductRelated | numeric | Number of product pages viewed |
| 6 | ProductRelated_Duration | numeric | Seconds spent on product pages |
| 7 | BounceRates | numeric | Average bounce rate of pages visited |
| 8 | ExitRates | numeric | Average exit rate of pages visited |
| 9 | PageValues | numeric | Average value of pages visited before a transaction |
| 10 | SpecialDay | numeric | Closeness of the visit to a special shopping day (0–1) |
| 11 | Month | categorical | Month of the visit |
| 12 | OperatingSystems | numeric | Encoded operating system identifier |
| 13 | Browser | numeric | Encoded browser identifier |
| 14 | Region | numeric | Encoded geographic region |
| 15 | TrafficType | numeric | Encoded traffic source |
| 16 | VisitorType | categorical | Returning, new, or other |
| 17 | Weekend | categorical | Whether the visit fell on a weekend |

**Preprocessing applied**
- Numeric: median imputation → standard scaling
- Categorical: most-frequent imputation → one-hot encoding (unknown categories ignored)
- Target: label encoded
- All steps are inside a scikit-learn `Pipeline`, so the saved models apply the identical
  transformations to any uploaded test CSV.

## c. GitHub repository link

<paste your repository URL here>

Live Streamlit app: <paste your Streamlit URL here>

## d. Models used

All five models were trained on the same training split and evaluated on the same
held-out test split (`test_data.csv`).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8896 | 0.8963 | 0.7734 | 0.4151 | 0.5402 | 0.5143 |
| Decision Tree | 0.8853 | 0.8861 | 0.6608 | 0.5472 | 0.5986 | 0.5356 |
| KNN | 0.8827 | 0.8395 | 0.7576 | 0.3669 | 0.4944 | 0.4738 |
| Naive Bayes | 0.7018 | 0.8192 | 0.3203 | 0.8092 | 0.4590 | 0.3649 |
| Support Vector Machine | 0.8932 | 0.8778 | 0.7382 | 0.4906 | 0.5894 | 0.5455 |

Metrics are computed on the 3,052-row held-out test split (`test_data.csv`) at the
default 0.5 decision threshold. Precision, recall and F1 are reported for the
positive class (`Revenue = True`).

**Majority-class baseline:** predicting `False` for every session scores 0.8437
accuracy with an MCC of 0.0. Every accuracy figure above must be read against that
floor — the strongest model beats it by only 4.95 percentage points, and Naive Bayes
falls 14.2 points below it. This is why the comparison below is argued on MCC and
recall rather than on accuracy.

**Hyper-parameters**

| Model | Settings |
|---|---|
| Logistic Regression | `max_iter=2000, C=1.0` |
| Decision Tree | `max_depth=9, min_samples_leaf=5` |
| KNN | `n_neighbors=12, weights='distance'` |
| Naive Bayes | `GaussianNB()` (default smoothing) |
| SVM | `kernel='rbf', C=1.0, probability=True` |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | <Does a linear boundary fit this data? Comment on the gap between accuracy and MCC, and on which class it fails.> |
| Decision Tree | <Did it overfit? Compare train vs test. Which depth helped? Which features dominated the splits?> |
| KNN | <Effect of scaling and of k. Does the class imbalance or dimensionality hurt it?> |
| Naive Bayes | <The feature-independence assumption is violated how badly here? Note its recall vs precision trade-off.> |
| Support Vector Machine | <RBF kernel behaviour, cost of training time, whether the margin handles the overlap well.> |
| **Overall winner for this dataset** | <Name the model and justify with MCC and AUC, not accuracy alone — especially if the classes are imbalanced. Mention the runner-up and the trade-off, e.g. "SVM wins on MCC (0.81) but takes 40× longer to train than Logistic Regression, which reaches 0.79.">|

---

## Repository structure

```
project-folder/
├── app.py                      # Streamlit application
├── train_models.py             # trains and saves all five models
├── requirements.txt
├── README.md
├── test_data.csv               # held-out test split used in the app
├── data/
│   └── raw_dataset.csv         # original dataset
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── support_vector_machine.joblib
│   ├── label_encoder.joblib
│   └── metadata.joblib
└── results/
    └── metrics_comparison.csv
```

## How to run locally

```bash
pip install -r requirements.txt
python train_models.py      # regenerates model/ and test_data.csv
streamlit run app.py
```

## App features

- CSV upload for test data
- Model selection dropdown (all five classifiers)
- Accuracy, AUC, Precision, Recall, F1 and MCC for the selected model
- Confusion matrix heatmap and full classification report
- A tab comparing all five models on the uploaded file
- Downloadable predictions

Models are loaded from disk and used for inference only — no retraining happens in the app.
