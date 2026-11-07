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

https://github.com/2026ak05619/online-shoppers-purchase-intent

https://purchase-intent-predictor.streamlit.app

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
| Logistic Regression | A linear boundary fits this data surprisingly well: it posts the highest AUC of all five models (0.8963), meaning it ranks sessions by purchase likelihood better than anything else here. The weakness is the decision rule, not the model. Recall is only 0.4151, so it misses roughly 58% of actual buyers, because at a 0.5 threshold the 84/16 class imbalance pushes almost every borderline session to the negative class. Accuracy of 0.8896 looks strong but sits just 4.6 points above the 0.8437 majority-class baseline. Lowering the threshold would very likely make this the strongest model on MCC. |
|---|---|
| Decision Tree | The best F1 of the five (0.5986) and the most balanced precision/recall pair (0.6608 / 0.5472). It is the only model that identifies more than half the buyers while keeping precision reasonable. Capping depth at 9 with a minimum of 5 samples per leaf kept it from memorising the training split, and because trees split on raw thresholds it needs no scaling at all. It also trains in under a second, against several minutes for the SVM. |
|---|---|
| KNN | The weakest of the four strong models, with the lowest MCC (0.4738) and an AUC of 0.8395. Recall of 0.3669 is the clearest symptom: after one-hot encoding Month and VisitorType the feature space exceeds 25 dimensions, and with only 15.63% positives, the 12 nearest neighbours of any given session are almost always non-buyers, so the distance-weighted vote rarely tips positive. Scaling was essential — without it the duration features, measured in thousands of seconds, would have dominated every distance calculation. |
|---|---|
| Naive Bayes | The clear outlier. Accuracy falls to 0.7018, below the 0.8437 baseline, and precision collapses to 0.3203, meaning roughly two false alarms for every real buyer. But recall jumps to 0.8092, the highest of any model. The cause is the independence assumption: BounceRates and ExitRates measure closely related behaviour, as do ProductRelated and ProductRelated_Duration, so correlated evidence is counted repeatedly and posteriors are driven toward extremes. For a use case where the cost of a wrongly offered discount is low, this recall-heavy profile may actually be the most useful behaviour. |
|---|---|
| Support Vector Machine | Wins on MCC (0.5455) and accuracy (0.8932), so on the metric least distorted by class imbalance it is the strongest model. The RBF kernel captures a non-linear boundary that Logistic Regression can only approximate, and it holds precision at 0.7382 while reaching higher recall (0.4906) than either LR or KNN. The cost is training time — by far the slowest of the five on 9,153 training rows, since it scales roughly quadratically with sample count and would become impractical on a larger dataset. |
|---|---|
| **Overall winner for this dataset** | Support Vector Machine, on MCC (0.5455), the metric that best survives an 84/16 split. But the margin is thin and the conclusion is qualified: Decision Tree trails by only 0.0099 MCC while beating SVM on F1 (0.5986 vs 0.5894) and recall (0.5472 vs 0.4906), trains in a fraction of the time, and can be read as explicit rules. If the business objective is catching as many buyers as possible, Decision Tree is the better choice. Logistic Regression's class-leading AUC (0.8963) further suggests that threshold tuning rather than model choice is where the largest gain remains. |

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
