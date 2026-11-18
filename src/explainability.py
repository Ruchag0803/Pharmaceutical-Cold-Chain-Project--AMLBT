"""SHAP explanations using the fitted model's transformed numeric inputs."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def _positive_class_values(shap_values, class_index=1):
    """Normalize SHAP binary-class output to a finite (rows, features) array."""
    raw = shap_values.values if hasattr(shap_values, "values") else shap_values
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim == 3:
        if raw.shape[-1] == 2:
            raw = raw[..., class_index]
        elif raw.shape[0] == 2:
            raw = raw[class_index, ...]
        else:
            raise ValueError(f"Cannot identify binary class axis in SHAP shape {raw.shape}")
    if raw.ndim != 2:
        raise ValueError(f"Expected 2-D SHAP values after class selection, got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("SHAP values contain NaN or infinite values")
    return raw


def explain(model_path, data, model_name, max_rows=300):
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is required for explanations; install requirements.txt") from exc

    root = Path(__file__).resolve().parents[1]
    model = joblib.load(model_path)
    if not hasattr(model, "steps") or len(model.steps) < 2:
        raise TypeError("Expected a fitted sklearn Pipeline with preprocessing and a classifier")
    preprocessor, estimator = model.steps[-2][1], model.steps[-1][1]
    raw_test = data["raw_test"].sample(min(max_rows, len(data["raw_test"])), random_state=42)
    raw_background = data["raw_train"].sample(min(80, len(data["raw_train"])), random_state=42)

    # This is the same dense numeric feature matrix that the estimator sees.
    feature_names = np.asarray(preprocessor.get_feature_names_out(), dtype=str)
    X = np.asarray(preprocessor.transform(raw_test), dtype=np.float64)
    background = np.asarray(preprocessor.transform(raw_background), dtype=np.float64)
    if X.ndim != 2 or background.ndim != 2 or X.shape[1] != len(feature_names):
        raise ValueError(f"Unexpected transformed shapes: X={X.shape}, background={background.shape}, features={len(feature_names)}")
    if not np.isfinite(X).all() or not np.isfinite(background).all():
        raise ValueError("Transformed SHAP inputs contain NaN or infinite values")

    explainer = shap.Explainer(estimator, background, feature_names=feature_names)
    raw_shap = explainer(X)
    values = _positive_class_values(raw_shap, class_index=1)
    if values.shape != X.shape:
        raise ValueError(f"SHAP values shape {values.shape} does not match transformed feature shape {X.shape}")
    base = np.asarray(raw_shap.base_values, dtype=np.float64)
    if base.ndim == 2:
        base = base[:, 1] if base.shape[-1] == 2 else base[:, 0]
    if base.ndim == 0:
        base = np.repeat(base, len(X))
    normalized = shap.Explanation(values=values, base_values=base, data=X, feature_names=feature_names)

    plot_dir = root / "outputs/plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    shap.plots.beeswarm(normalized, show=False, max_display=20)
    plt.tight_layout(); plt.savefig(plot_dir / "shap_summary.png", dpi=160, bbox_inches="tight"); plt.close()
    shap.plots.bar(normalized, show=False, max_display=20)
    plt.tight_layout(); plt.savefig(plot_dir / "shap_feature_importance.png", dpi=160, bbox_inches="tight"); plt.close()

    importance = pd.DataFrame({"feature": feature_names, "mean_abs_shap": np.abs(values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(root / "outputs/metrics/shap_feature_importance.csv", index=False)
    for feature in importance.head(3)["feature"]:
        shap.plots.scatter(normalized[:, feature], color=normalized, show=False)
        plt.tight_layout(); plt.savefig(plot_dir / f"shap_dependence_{feature.replace('/', '_')}.png", dpi=160, bbox_inches="tight"); plt.close()
    shap.plots.waterfall(normalized[0], show=False, max_display=15)
    plt.tight_layout(); plt.savefig(plot_dir / "shap_individual_waterfall.png", dpi=160, bbox_inches="tight"); plt.close()

    table = "| feature | mean absolute SHAP |\n|---|---:|\n" + "\n".join(
        f"| {row.feature} | {row.mean_abs_shap:.6f} |" for row in importance.head(15).itertuples()
    )
    report = (
        "# SHAP Explainability Report\n\n"
        "SHAP values describe the fitted model's predictions and are not causal effects. The explainer received the transformed numeric matrix and original one-hot feature names from the fitted preprocessing pipeline. Potentially post-outcome temperature summaries and `Temperature_Exposure_Risk` were excluded from predictors.\n\n"
        "## Leading features\n\n" + table + "\n\n"
        "Summary, feature importance, dependence, and individual waterfall plots are saved under `outputs/plots/`. Attribution direction depends on the model and correlated inputs.\n"
    )
    (root / "reports/SHAP_Explainability_Report.md").write_text(report, encoding="utf-8")
    return importance
