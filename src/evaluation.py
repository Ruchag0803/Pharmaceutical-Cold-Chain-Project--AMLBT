"""Shared binary classification metrics and plots."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, classification_report, RocCurveDisplay, PrecisionRecallDisplay)

def evaluate(name, model, X, y, out_dir, split="test"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    scores = {"model": name, "split": split, "accuracy": accuracy_score(y,pred), "precision": precision_score(y,pred,zero_division=0), "recall": recall_score(y,pred,zero_division=0), "f1": f1_score(y,pred,zero_division=0), "roc_auc": roc_auc_score(y,prob) if prob is not None else None, "pr_auc": average_precision_score(y,prob) if prob is not None else None}
    (out / f"{name}_{split}_metrics.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    (out / f"{name}_{split}_classification_report.txt").write_text(classification_report(y,pred,zero_division=0), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(5,4)); sns.heatmap(confusion_matrix(y,pred), annot=True, fmt="d", cmap="Blues", ax=ax); ax.set(xlabel="Predicted", ylabel="Actual", title=f"{name} {split} confusion matrix"); fig.tight_layout(); fig.savefig(out / f"{name}_{split}_confusion.png", dpi=150); plt.close(fig)
    return scores

def save_curves(name, model, X, y, out_dir):
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    if not hasattr(model,"predict_proba"): return
    fig,ax=plt.subplots(figsize=(5,4)); RocCurveDisplay.from_estimator(model,X,y,ax=ax); fig.tight_layout(); fig.savefig(out/f"{name}_roc.png",dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4)); PrecisionRecallDisplay.from_estimator(model,X,y,ax=ax); fig.tight_layout(); fig.savefig(out/f"{name}_pr.png",dpi=150); plt.close(fig)
