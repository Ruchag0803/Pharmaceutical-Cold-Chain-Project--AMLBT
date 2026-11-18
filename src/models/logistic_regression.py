from sklearn.linear_model import LogisticRegression
from src.models._common import run_model
def train(data=None): return run_model("logistic_regression", LogisticRegression(max_iter=2000,class_weight="balanced",random_state=42), True, data)
if __name__ == "__main__": train()
