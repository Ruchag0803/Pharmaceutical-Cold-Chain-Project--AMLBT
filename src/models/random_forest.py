from sklearn.ensemble import RandomForestClassifier
from src.models._common import run_model
def train(data=None): return run_model("random_forest", RandomForestClassifier(n_estimators=300,max_depth=None,min_samples_leaf=3,class_weight="balanced",n_jobs=-1,random_state=42), False, data)
if __name__ == "__main__": train()
