from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
def save_target_plot(df, path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); fig,ax=plt.subplots(figsize=(5,4)); sns.countplot(data=df,x="Excursion_Before_Delivery",ax=ax); ax.set(title="Excursion target distribution",xlabel="Excursion before delivery",ylabel="Shipments"); fig.tight_layout(); fig.savefig(p,dpi=150); plt.close(fig)
