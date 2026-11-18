from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from src.preprocessing import load_dataset, split_data
ROOT=Path(__file__).resolve().parents[1]

MODEL_NAMES = [
    "logistic_regression", "naive_bayes", "decision_tree", "random_forest",
    "xgboost", "lightgbm", "svm",
]

def plot_combined_curves():
    """Plot all saved models on the same deterministic held-out test split."""
    data = split_data(load_dataset())
    y_test = data["y_test"].to_numpy(dtype="int64")
    model_paths = {name: ROOT / "outputs/saved_models" / f"{name}.joblib" for name in MODEL_NAMES}
    missing = [name for name, path in model_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing saved models required for combined curves: {', '.join(missing)}")

    plot_dir = ROOT / "outputs/plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    predictions = {}
    for name in MODEL_NAMES:
        model = joblib.load(model_paths[name])
        if not hasattr(model, "predict_proba"):
            raise TypeError(f"Saved model {name} does not provide predict_proba")
        probabilities = model.predict_proba(data["raw_test"])[:, 1]
        predictions[name] = probabilities

    # Ensure curves and saved metric labels refer to the same test predictions.
    for name, probabilities in predictions.items():
        metric_path = ROOT / "outputs/metrics" / f"{name}_test_metrics.json"
        if not metric_path.exists():
            raise FileNotFoundError(f"Missing held-out test metrics for {name}: {metric_path}")
        recorded = json.loads(metric_path.read_text(encoding="utf-8"))
        roc_auc = roc_auc_score(y_test, probabilities)
        pr_auc = average_precision_score(y_test, probabilities)
        if abs(roc_auc - recorded["roc_auc"]) > 1e-10 or abs(pr_auc - recorded["pr_auc"]) > 1e-10:
            raise ValueError(f"Recreated test predictions for {name} do not match saved test metrics; refusing to plot mismatched splits")

    fig, ax = plt.subplots(figsize=(10, 7))
    for name in MODEL_NAMES:
        probs = predictions[name]
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name.replace('_', ' ').title()} (ROC-AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2, label="Chance")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curves — Held-out Test Set")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(plot_dir / "combined_roc_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    for name in MODEL_NAMES:
        probs = predictions[name]
        precision, recall, _ = precision_recall_curve(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        ax.plot(recall, precision, linewidth=2, label=f"{name.replace('_', ' ').title()} (PR-AUC = {pr_auc:.3f})")
    prevalence = y_test.mean()
    ax.axhline(prevalence, linestyle="--", color="gray", linewidth=1.2, label=f"Positive-class prevalence = {prevalence:.3f}")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision", title="Precision–Recall Curves — Held-out Test Set")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(plot_dir / "combined_precision_recall_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

def compare():
    files=list((ROOT/"outputs/metrics").glob("*_test_metrics.json"))
    if not files: raise FileNotFoundError("No test metric files found. Run main.py first.")
    df=pd.DataFrame([json.loads(p.read_text(encoding="utf-8")) for p in files])
    val_files=list((ROOT/"outputs/metrics").glob("*_validation_metrics.json"))
    val=pd.DataFrame([json.loads(p.read_text(encoding="utf-8")) for p in val_files])
    rank=val.set_index("model")[["recall","pr_auc","f1"]].rename(columns=lambda c:f"validation_{c}")
    df=df.merge(rank,left_on="model",right_index=True,how="left").sort_values(["validation_recall","validation_pr_auc","validation_f1"],ascending=False)
    df.to_csv(ROOT/"outputs/metrics/model_comparison.csv",index=False)
    ax=df.set_index("model")[["accuracy","precision","recall","f1","roc_auc","pr_auc"]].plot(kind="bar",figsize=(12,6)); ax.set_ylim(0,1); ax.set_ylabel("Score"); plt.tight_layout(); plt.savefig(ROOT/"outputs/plots/model_comparison.png",dpi=150); plt.close()
    def markdown_table(frame):
        cols=["model","accuracy","precision","recall","f1","roc_auc","pr_auc"]
        rows=[[str(x) if isinstance(x,str) else (f"{x:.4f}" if pd.notna(x) else "") for x in row] for row in frame[cols].itertuples(index=False,name=None)]
        return "| " + " | ".join(cols) + " |\n|" + "|".join(["---"]*len(cols))+"|\n"+"\n".join("| "+" | ".join(r)+" |" for r in rows)
    lines=["# Model Comparison Report", "", "Final metrics below use the held-out test set. Candidate ranking uses validation recall, then validation PR-AUC and F1.", "", markdown_table(df), ""]
    for key,label in [("recall","Highest recall"),("f1","Best F1"),("roc_auc","Best ROC-AUC"),("pr_auc","Best PR-AUC")]:
        best=df.loc[df[key].idxmax()]; lines.append(f"- {label}: **{best['model']}** ({key}={best[key]:.4f}).")
    lines += ["", f"Validation-selected candidate: **{df.iloc[0]['model']}**. Review false negatives and operational alert capacity before deployment.", "", "A false negative is an excursion shipment labeled safe. A single holdout split is not prospective validation."]
    (ROOT/"reports/Model_Comparison_Report.md").write_text("\n".join(lines),encoding="utf-8")
    plot_combined_curves()
    return df
if __name__=="__main__": print(compare().to_string(index=False))
