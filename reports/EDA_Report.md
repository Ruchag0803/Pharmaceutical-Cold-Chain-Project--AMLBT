# EDA Report

## Dataset overview

The supplied CSV contains 5,000 shipment records and 38 columns. The target is `Excursion_Before_Delivery`; it has 4,384 zeros and 616 ones (12.32% positives). `Medication_Type`, product minimum and maximum temperatures are constant in this file. There are 0 missing values and 0 exact duplicate rows.

## Data quality and unusual values

The target contains both classes and only values 0 and 1. Most fields are complete. Product temperatures are fixed at 2°C and 8°C, limiting their discriminative value. The sample has 181 distinct shipment dates and no unique identifier duplicates. Numeric ranges should be checked against domain-approved operating limits before decisions are made; this dataset alone does not define invalid thresholds. The CSV contains a visibly garbled character in one coolant label, a text-encoding quality issue that should be corrected at source when possible.

## Target and class imbalance

Positives are 12.32% (616/5,000), versus 87.68% non-excursions. This is a material imbalance: accuracy can look strong while missing many excursions. Recall, PR-AUC, precision, F1, and the confusion matrix should be reviewed together.

## Feature relationships and leakage

The supplied columns include ambient conditions, insulation and coolant attributes, sensor flags, temperature summaries, trend, threshold proximity, and an exposure-risk category. Temperature summaries, trend/proximity, and `Temperature_Exposure_Risk` may be computed from observations accumulated during transit or directly encode the outcome. They are excluded from model predictors pending confirmation of prediction-time availability. Shipment ID and date are also excluded. Operational predictors such as ambient temperature, insulation, coolant quantity, delivery distance, sensor flags, and refrigeration status remain candidates if they are available at the intended prediction moment.

## Conclusions

The target is valid and the dataset is complete, but the class imbalance and timing ambiguity are important. Establish a precise prediction timestamp and validate feature provenance before model use. The notebook provides reproducible univariate distributions, target plots, and grouped feature comparisons; descriptive patterns are not causal evidence.
