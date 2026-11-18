from sklearn.tree import DecisionTreeClassifier
from src.models._common import run_model
def train(data=None): return run_model("decision_tree", DecisionTreeClassifier(max_depth=6,min_samples_split=20,min_samples_leaf=10,class_weight="balanced",random_state=42), False, data)
if __name__ == "__main__": train()
