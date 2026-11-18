from pathlib import Path
import json
import os
import subprocess
import sys
os.environ.setdefault("MPLBACKEND", "Agg")
from src.preprocessing import load_dataset, validate_dataset, split_data
from src.models.logistic_regression import train as train_lr
from src.models.naive_bayes import train as train_nb
from src.models.decision_tree import train as train_dt
from src.models.random_forest import train as train_rf
from src.models.xgboost_model import train as train_xgb
from src.models.lightgbm_model import train as train_lgbm
from src.models.svm import train as train_svm
from src.compare_models import compare
from src.explainability import explain

def main():
    root=Path(__file__).resolve().parent
    df=load_dataset(); validate_dataset(df); data=split_data(df)
    # Run LightGBM before XGBoost: their Windows native OpenMP runtimes can
    # interfere when both libraries train in the same process.
    runners=[("logistic_regression",train_lr),("naive_bayes",train_nb),("decision_tree",train_dt),("random_forest",train_rf),("lightgbm",train_lgbm),("xgboost",train_xgb),("svm",train_svm)]
    failures=[]
    for name,runner in runners:
        try:
            print(f"Training {name}...")
            if name == "lightgbm":
                # LightGBM's Windows native runtime can fault after sklearn's
                # OpenMP runtime has been initialized in this process. Run its
                # module in a clean process; fixed seed and shared split code
                # reproduce the identical train/validation/test membership.
                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = "1"
                subprocess.run(
                    [sys.executable, "-m", "src.models.lightgbm_model"],
                    cwd=root, env=env, check=True,
                )
            else:
                runner(data)
        except Exception as exc:
            failures.append({"model":name,"error":str(exc)}); print(f"Could not run {name}: {exc}")
    if not list((root/"outputs/metrics").glob("*_test_metrics.json")):
        raise RuntimeError("No models produced results")
    results=compare()
    selected=results.iloc[0]["model"]
    selected_path=root/"outputs/saved_models"/f"{selected}.joblib"
    # SHAP is a required stage; failures propagate with a traceback instead
    # of being reported as a successful run after silently skipping it.
    explain(selected_path,data,selected)
    issues_path = root / "outputs/metrics/run_issues.json"
    if failures:
        issues_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    else:
        issues_path.unlink(missing_ok=True)
    print(f"Selected by validation recall, PR-AUC, then F1: {selected}")
    if failures: print("Models requiring attention:",json.dumps(failures))
if __name__=="__main__": main()
