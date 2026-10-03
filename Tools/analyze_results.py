# tools/analyze_results.py
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

CSV_PATH = os.path.join(BASE_DIR, "data", "embeddings", "distances_arcface.csv")
OUT_REPORT = os.path.join(BASE_DIR, "data", "embeddings", "results_report.txt")

# embedding ها
EMB_DIR = os.path.join(BASE_DIR, "data", "embeddings")
CLEAN_EMB = os.path.join(EMB_DIR, "clean_arcface.npy")
ATTACK_EMB = os.path.join(EMB_DIR, "attack_arcface.npy")
DEFENSE_EMB = os.path.join(EMB_DIR, "defense_arcface.npy")

HIST_OUT = os.path.join(EMB_DIR, "histogram_arcface_clean_attack_defense.png")

df = pd.read_csv(CSV_PATH)

# ----------------------------------
# محاسبه اشتباهات حمله و دفاع
# ----------------------------------

threshold = df["dist_clean_attack"].median()
df["attack_wrong"] = df["dist_clean_attack"] > threshold
df["defense_fixed"] = df["dist_attack_defense"] < df["dist_clean_attack"]

# ----------------------------------
# accuracy ها
# ----------------------------------

clean_acc = 1.0
attack_acc = 1 - df["attack_wrong"].mean()
defense_acc = df["defense_fixed"].mean()

# ----------------------------------
# success rates
# ----------------------------------

ASR = df["attack_wrong"].mean()
DSR = (df["attack_wrong"] & df["defense_fixed"]).mean()

# ----------------------------------
# میانگین فاصله‌ها
# ----------------------------------

mean_dist = df[[
    "dist_clean_attack",
    "dist_clean_defense",
    "dist_attack_defense"
]].mean()

# ----------------------------------
# ذخیره گزارش متنی
# ----------------------------------

with open(OUT_REPORT, "w", encoding="utf-8") as f:
    f.write("=== RESULT ANALYSIS ===\n\n")
    f.write(f"Clean Accuracy:   {clean_acc:.4f}\n")
    f.write(f"Attack Accuracy:  {attack_acc:.4f}\n")
    f.write(f"Defense Accuracy: {defense_acc:.4f}\n\n")

    f.write(f"Attack Success Rate (ASR):  {ASR:.4f}\n")
    f.write(f"Defense Success Rate (DSR): {DSR:.4f}\n\n")

    f.write("Mean Distances:\n")
    f.write(str(mean_dist))
    f.write("\n")

print("Text report saved to:", OUT_REPORT)

# =====================================================
# رسم هیستوگرام تجمیعی embedding ها
# =====================================================

clean_emb = np.load(CLEAN_EMB).ravel()
attack_emb = np.load(ATTACK_EMB).ravel()
defense_emb = np.load(DEFENSE_EMB).ravel()

plt.figure(figsize=(7, 5))

plt.hist(
    clean_emb,
    bins=80,
    alpha=0.5,
    label="Clean",
    density=True
)

plt.hist(
    attack_emb,
    bins=80,
    alpha=0.5,
    label="Attack",
    density=True
)

plt.hist(
    defense_emb,
    bins=80,
    alpha=0.5,
    label="Defense (Final)",
    density=True
)

plt.xlabel("Embedding value")
plt.ylabel("Density")
plt.title("Histogram of ArcFace Embeddings (Clean / Attack / Defense)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(HIST_OUT, dpi=300)
plt.close()

print("Histogram saved to:", HIST_OUT)
print("Done.")




#برای ران هر دو تای زیر رو با هم و همزمان بزن
#pip install pandas
#pip install numpy==1.26.4

#بعدش بزن
#cd D:\PayanNameh\paper-structure\deep-keypoints-attack
#.\venv\Scripts\activate
#python tools\analyze_results.py
