def train(data=None):
    try: from xgboost import XGBClassifier
    except ImportError as e: raise RuntimeError("XGBoost is required; install requirements.txt") from e
    from src.models._common import run_model
    return run_model("xgboost", XGBClassifier(n_estimators=300,max_depth=4,learning_rate=.05,subsample=.85,colsample_bytree=.85,eval_metric="logloss",random_state=42,n_jobs=-1), False, data)
if __name__ == "__main__": train()
