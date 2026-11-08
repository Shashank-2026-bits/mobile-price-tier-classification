# Mobile Price Tier Classification – ML Assignment 2

Live app: https://mobile-price-tier-classification-2026aj05972.streamlit.app/

## a. Problem Statement

A phone maker wants to estimate the price segment of a new handset from its hardware specifications. Given 20 technical attributes of a mobile phone (RAM, battery capacity, screen resolution, camera megapixels, connectivity, etc.), the task is to predict its **price range** as one of four classes: *0 = Low cost, 1 = Medium cost, 2 = High cost, 3 = Very high cost*. This is a **multi-class classification** problem. Five classifiers are trained, evaluated, and compared, and a Streamlit app is used to evaluate the saved models on held-out test data.

## b. Dataset Description

| Item | Detail |
|---|---|
| Name | Mobile Price Classification |
| Source | Kaggle – https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification (`train.csv`; Kaggle's `test.csv` has no labels, so it is not used) |
| Instances | 2,000 |
| Features | 20 (all numeric) |
| Target | `price_range` – 4 classes, perfectly balanced (500 each) |
| Missing values | None |
| Train / test split | 80 / 20 stratified (1,600 / 400), `random_state=42`; the 400 test rows are in `test_data.csv` |

**Features**

| Type | Columns |
|---|---|
| Binary (0/1) | `blue`, `dual_sim`, `four_g`, `three_g`, `touch_screen`, `wifi` |
| Continuous / count | `battery_power` (mAh), `clock_speed` (GHz), `fc` (front camera MP), `pc` (primary camera MP), `int_memory` (GB), `m_dep` (depth cm), `mobile_wt` (g), `n_cores`, `px_height`, `px_width` (resolution), `ram` (MB), `sc_h`, `sc_w` (screen cm), `talk_time` (h) |

**Observations from EDA**
- `ram` dominates: correlation with `price_range` is **0.917**; the next are `battery_power` (0.201), `px_width` (0.166), `px_height` (0.149). The remaining features are nearly uncorrelated with price.
- `sc_w` is 0 in 180 rows and `px_height` is 0 in 2 rows (likely data-entry artefacts); they were kept as-is because they carry little signal.

**Preprocessing**: no encoding needed (all numeric). Each model is saved as a scikit-learn `Pipeline`; scale-sensitive models (Logistic Regression, KNN, SVM) include a `StandardScaler` inside the pipeline, so the app can feed raw test-CSV columns directly. Hyperparameters were tuned with 5-fold stratified `GridSearchCV` (macro-F1) on the training split only.

## c. GitHub Repository Link

https://github.com/Shashank-2026-bits/mobile-price-tier-classification

```
├── app.py                 # Streamlit app (loads saved models, no retraining)
├── train_models.py        # Training, tuning, evaluation, saving
├── requirements.txt
├── .streamlit/config.toml # App theme
├── test_data.csv          # 400-row held-out test set (features + price_range)
├── data/mobile_price.csv  # Original Kaggle train.csv
└── model/
    ├── logistic_regression.pkl, decision_tree.pkl, knn.pkl, naive_bayes.pkl, svm.pkl
    ├── metrics_summary.csv  # Test-set metrics
    └── model_info.json      # Feature list, class names, tuned hyperparameters
```

Run locally:
```
pip install -r requirements.txt
python train_models.py        # optional – regenerates model/ and test_data.csv
streamlit run app.py
```

## d. Models Used

| Model | Implementation | Tuned hyperparameters |
|---|---|---|
| Logistic Regression | `StandardScaler` + `LogisticRegression` | C = 100 |
| Decision Tree | `DecisionTreeClassifier` | max_depth = 6, min_samples_leaf = 5 |
| KNN | `StandardScaler` + `KNeighborsClassifier` | k = 31, weights = distance |
| Naive Bayes | `GaussianNB` | var_smoothing = 1e-5 |
| Support Vector Machine | `StandardScaler` + `CalibratedClassifierCV(SVC)` | kernel = linear, C = 10 |

Metrics are computed on the 400-row test set. Precision, Recall and F1 are **macro-averaged**; AUC is **one-vs-rest macro** AUC from predicted probabilities.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9992 | 0.9826 | 0.9825 | 0.9825 | 0.9767 |
| Decision Tree | 0.8525 | 0.9625 | 0.8590 | 0.8525 | 0.8530 | 0.8052 |
| KNN | 0.6400 | 0.8430 | 0.6480 | 0.6400 | 0.6430 | 0.5206 |
| Naive Bayes | 0.8050 | 0.9526 | 0.8056 | 0.8050 | 0.8053 | 0.7400 |
| Support Vector Machine | 0.9750 | 0.9924 | 0.9750 | 0.9750 | 0.9750 | 0.9667 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best model: 98.25% accuracy, MCC 0.977, only 7 of 400 phones misclassified and never by more than one tier. The price tiers are close to linearly separable in the scaled feature space (driven mainly by `ram`, then battery and resolution), so a linear decision boundary fits well. Weak regularisation (C = 100) was preferred, showing the model was not overfitting. |
| Decision Tree | Moderate (85.25% accuracy, MCC 0.805). The tree relies on `ram` for 72% of its importance and approximates the smooth linear boundaries with axis-aligned steps, so phones near tier boundaries are often misplaced, especially the middle tiers (Medium: 74/100 correct). Depth was capped at 6 to control overfitting; deeper trees did not improve CV score. |
| KNN | Weakest (64% accuracy, MCC 0.521) and the only model with errors of two tiers. After scaling, all 20 features carry equal weight in the distance, and the ~16 irrelevant ones drown out `ram`. A check with only the top-4 correlated features gives ~90% CV accuracy, confirming the curse-of-dimensionality / irrelevant-feature problem rather than an unsuitable dataset. |
| Naive Bayes | 80.5% accuracy, MCC 0.740. Fast and with a good AUC (0.953), so its probability ranking is reasonable, but the independence assumption ignores correlated pairs (`px_height`/`px_width`, `fc`/`pc`, `three_g`/`four_g`) and the Gaussian assumption does not fit the binary features, which hurts the two middle classes most. |
| Support Vector Machine | Very close second (97.5% accuracy, MCC 0.967). Grid search selected a **linear** kernel over RBF, agreeing with the Logistic Regression result that the classes are linearly separable. Calibration was needed to produce probabilities for AUC; it costs more training time than Logistic Regression for no gain in accuracy. |
| Overall winner for this dataset | **Logistic Regression** – highest Accuracy, Precision, Recall, F1 and MCC, with AUC 0.999 (on par with SVM). It is also the simplest, fastest and most interpretable model. Linear models clearly outperform distance- and tree-based models here because price tier is close to a linear function of RAM and a few other specifications. |

## Streamlit App Features

The app uses a Microsoft Fluent 2 light theme (Segoe UI, `.streamlit/config.toml` tokens).

- **Data source** (navigation pane) – evaluate the bundled `test_data.csv` straight away, or switch to **Upload CSV** for any CSV with the 20 features + `price_range`; a sample can be downloaded from the app.
- **Model selection dropdown** – choose any of the five saved models; tuned hyperparameters are shown.
- **KPI cards** – Accuracy, AUC, Precision, Recall, F1, MCC, each compared with the average of the other models.
- **Overview** – confusion matrix and per-tier classification report for the selected model.
- **Model comparison** – metric table (best values highlighted) and a grouped bar chart with selectable metrics.
- **Predictions** – row-level results with confidence, filter to correct/misclassified rows, CSV export.
- **Data explorer** – tier distribution, feature-by-tier box plot, summary statistics and raw data.
- Input validation (missing columns, non-numeric or invalid target values). Models are only loaded – never retrained – in the app.
