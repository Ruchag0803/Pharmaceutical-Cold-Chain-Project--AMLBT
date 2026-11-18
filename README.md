# Pharmaceutical Cold-Chain Excursion Prediction

Predict `Excursion_Before_Delivery` from shipment attributes using seven classifiers and SHAP explanations. The source CSV is preserved in `data/`.

## Setup

From this directory, create/activate a Python environment and run `python -m pip install -r requirements.txt`. Then run `python main.py`. Individual classifiers can run as modules, for example `python -m src.models.logistic_regression`. The EDA notebook is `notebooks/01_EDA.ipynb`.

Outputs are written to `outputs/metrics`, `outputs/plots`, and `outputs/saved_models`; reports are in `reports/`.

## Data and modeling notes

The target is binary (0/1). A stratified 70/15/15 train/validation/test split with seed 42 is shared across models. Preprocessing is fitted within each model's training pipeline. Shipment IDs, dates, `Temperature_Exposure_Risk`, and observed temperature summaries/trends/proximity are excluded because they may encode post-dispatch or outcome information. Confirm the operational prediction timestamp and feature availability before interpreting results. No resampling is automatic. Model comparison prioritizes class 1 recall and PR-AUC; review false negatives and alert burden before operational use.

## Limitations

This is observational prediction, not causal inference. The test metrics are a single holdout estimate. SHAP explanations are associations with the fitted model. Verify all input fields are available at the intended prediction time, especially sensor and refrigeration status fields.
