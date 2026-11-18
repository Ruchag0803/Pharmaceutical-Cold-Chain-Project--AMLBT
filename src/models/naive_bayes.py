from sklearn.naive_bayes import GaussianNB
from src.models._common import run_model
def train(data=None): return run_model("naive_bayes", GaussianNB(), True, data)
if __name__ == "__main__": train()
