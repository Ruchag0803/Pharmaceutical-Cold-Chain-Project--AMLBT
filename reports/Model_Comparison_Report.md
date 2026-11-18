# Model Comparison Report

Final metrics below use the held-out test set. Candidate ranking uses validation recall, then validation PR-AUC and F1.

| model | accuracy | precision | recall | f1 | roc_auc | pr_auc |
|---|---|---|---|---|---|---|
| logistic_regression | 0.9000 | 0.5612 | 0.8478 | 0.6753 | 0.9281 | 0.6640 |
| naive_bayes | 0.9027 | 0.5674 | 0.8696 | 0.6867 | 0.8889 | 0.4584 |
| svm | 0.9200 | 0.6270 | 0.8587 | 0.7248 | 0.9360 | 0.7975 |
| decision_tree | 0.9253 | 0.6500 | 0.8478 | 0.7358 | 0.9110 | 0.6594 |
| random_forest | 0.9413 | 0.7353 | 0.8152 | 0.7732 | 0.9460 | 0.8281 |
| lightgbm | 0.9453 | 0.7865 | 0.7609 | 0.7735 | 0.9477 | 0.7894 |
| xgboost | 0.9427 | 0.7882 | 0.7283 | 0.7571 | 0.9475 | 0.8160 |

- Highest recall: **naive_bayes** (recall=0.8696).
- Best F1: **lightgbm** (f1=0.7735).
- Best ROC-AUC: **lightgbm** (roc_auc=0.9477).
- Best PR-AUC: **random_forest** (pr_auc=0.8281).

Validation-selected candidate: **logistic_regression**. Review false negatives and operational alert capacity before deployment.

A false negative is an excursion shipment labeled safe. A single holdout split is not prospective validation.