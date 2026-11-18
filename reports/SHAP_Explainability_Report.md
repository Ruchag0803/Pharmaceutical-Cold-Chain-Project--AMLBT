# SHAP Explainability Report

SHAP values describe the fitted model's predictions and are not causal effects. The explainer received the transformed numeric matrix and original one-hot feature names from the fitted preprocessing pipeline. Potentially post-outcome temperature summaries and `Temperature_Exposure_Risk` were excluded from predictors.

## Leading features

| feature | mean absolute SHAP |
|---|---:|
| Refrigeration_Status_Normal | 0.950309 |
| Ambient_Temperature_C | 0.584679 |
| Refrigeration_Status_Reduced Performance | 0.564078 |
| Thermal_Holding_Capacity_Min | 0.318263 |
| Weather_Condition_Sunny | 0.307929 |
| Insulation_Level_Good | 0.219320 |
| Weather_Condition_Cloudy | 0.214744 |
| Estimated_Remaining_Transit_Time_min | 0.205840 |
| Planned_Delivery_Time_min | 0.205840 |
| Weather_Condition_Rainy | 0.171628 |
| Coolant_Type_Hybrid gel and PCM | 0.132977 |
| Coolant_Type_Water-based gel packs | 0.127846 |
| Power_Failure_Flag | 0.125149 |
| Insulation_Level_Poor | 0.124152 |
| Previous_Excursion_History | 0.097482 |

Summary, feature importance, dependence, and individual waterfall plots are saved under `outputs/plots/`. Attribution direction depends on the model and correlated inputs.
