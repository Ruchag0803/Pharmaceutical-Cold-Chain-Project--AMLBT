from sklearn.svm import SVC
from src.models._common import run_model
def train(data=None): return run_model("svm", SVC(C=1.0,kernel="rbf",probability=True,class_weight="balanced",random_state=42), True, data)
if __name__ == "__main__": train()
