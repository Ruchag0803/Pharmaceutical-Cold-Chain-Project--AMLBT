from pathlib import Path
import joblib
from sklearn.pipeline import make_pipeline
from src.preprocessing import load_dataset, split_data
from src.evaluation import evaluate, save_curves

def run_model(name, estimator, scale=True, data=None):
    root=Path(__file__).resolve().parents[2]
    data=split_data(data["_raw_df"], scale=scale) if data is not None else split_data(load_dataset(), scale=scale)
    model=make_pipeline(data["preprocessor"], estimator)
    model.fit(data["raw_train"], data["y_train"])
    metrics= evaluate(name,model,data["raw_val"],data["y_val"],root/"outputs/metrics", "validation")
    metrics.update(evaluate(name,model,data["raw_test"],data["y_test"],root/"outputs/metrics", "test"))
    save_curves(name,model,data["raw_test"],data["y_test"],root/"outputs/plots")
    (root/"outputs/saved_models").mkdir(parents=True,exist_ok=True)
    joblib.dump(model,root/"outputs/saved_models"/f"{name}.joblib")
    return model, metrics
