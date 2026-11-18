def train(data=None):
    try: from lightgbm import LGBMClassifier
    except ImportError as e: raise RuntimeError("LightGBM is required; install requirements.txt") from e
    from src.models._common import run_model
    # Use a deterministic single-thread CPU learner to avoid the parallel
    # native path implicated by access violations in some Windows builds.
    estimator = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=-1,
        class_weight="balanced", verbosity=-1, random_state=42,
        n_jobs=1, device_type="cpu", deterministic=True,
        force_col_wise=True,
    )
    return run_model("lightgbm", estimator, False, data)
if __name__ == "__main__": train()
