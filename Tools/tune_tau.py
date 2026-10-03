# tools/tune_tau.py
import pandas as pd
from sklearn.metrics import roc_curve

df = pd.read_csv("data/defense_stats/squeeze_summary.csv")

labels = pd.read_csv("data/defense_stats/labels.csv")
labels.columns = [c.strip() for c in labels.columns]
labels["label"] = labels["label"].astype(int)

# 🔹 حذف _squeeze تا merge درست انجام بشه
df["name"] = df["name"].str.replace("_squeeze", "", regex=False)
labels["name"] = labels["name"].str.replace("_squeeze", "", regex=False)

df = df.merge(labels, on="name")

fpr, tpr, thr = roc_curve(df["label"], df["max_div"])
cand = [(f, t) for f, t in zip(fpr, thr) if f <= 0.05]
if cand:
    chosen = cand[-1][1]
else:
    youden = tpr - fpr
    chosen = thr[youden.argmax()]

print("recommended_tau:", chosen)

pd.DataFrame({"fpr": fpr, "tpr": tpr, "thr": thr}).to_csv(
    "data/defense_stats/roc_squeeze.csv", index=False
)




# cd D:\PayanNameh\paper-structure\deep-keypoints-attack
#python tools/tune_tau.py



