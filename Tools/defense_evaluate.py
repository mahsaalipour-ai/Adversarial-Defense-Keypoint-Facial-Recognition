import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(__file__))

CSV_PATH = os.path.join(BASE, "data", "embeddings", "distances_arcface.csv")
OUT_REPORT = os.path.join(BASE, "data", "results", "defense_eval.txt")
OUT_PLOT = os.path.join(BASE, "data", "results", "defense_histogram.png")

# ----------------------------
# ۱) خواندن داده‌ها
# ----------------------------
df = pd.read_csv(CSV_PATH)

# ستون‌ها:
# dist_clean_attack
# dist_clean_defense
# dist_attack_defense

# ----------------------------
# ۲) تعیین threshold دفاعی
# ----------------------------
tau = df["dist_clean_defense"].median()

# ----------------------------
# ۳) برچسب‌دهی حمله و دفاع
# ----------------------------
df["attack_wrong"] = df["dist_clean_attack"] > df["dist_clean_attack"].median()
df["defense_fixed"] = df["dist_attack_defense"] < tau

# ----------------------------
# ۴) محاسبه نرخ‌ها
# ----------------------------
clean_acc = 1.0  
attack_acc = 1 - df["attack_wrong"].mean()
defense_acc = df["defense_fixed"].mean()

ASR = df["attack_wrong"].mean()
DSR = (df["attack_wrong"] & df["defense_fixed"]).mean()

# ----------------------------
# ۵) ذخیره گزارش
# ----------------------------
with open(OUT_REPORT, "w") as f:
    f.write("=== DEFENSE EVALUATION REPORT ===\n\n")
    f.write(f"tau (threshold): {tau:.4f}\n\n")

    f.write(f"Attack Success Rate (ASR): {ASR:.4f}\n")
    f.write(f"Defense Success Rate (DSR): {DSR:.4f}\n")
    f.write(f"Defense Accuracy: {defense_acc:.4f}\n\n")

    f.write("Mean Distances:\n")
    f.write(str(df[["dist_clean_attack","dist_clean_defense","dist_attack_defense"]].mean()))
    f.write("\n")

print("Saved:", OUT_REPORT)

# ----------------------------
# ۶) رسم هیستوگرام
# ----------------------------

# داده‌ها برای رسم هیستوگرام
distances = [df["dist_clean_attack"], df["dist_clean_defense"], df["dist_attack_defense"]]
labels = ["Clean - Attack", "Clean - Defense", "Attack - Defense"]

# رسم هیستوگرام
plt.figure(figsize=(8, 6))

# رسم هر سه هیستوگرام
plt.hist(distances[0], bins=50, alpha=0.5, label="Clean - Attack")
plt.hist(distances[1], bins=50, alpha=0.5, label="Clean - Defense")
plt.hist(distances[2], bins=50, alpha=0.5, label="Attack - Defense")

plt.title("Histogram of ArcFace Embeddings (Clean / Attack / Defense)", fontsize=14)
plt.xlabel("Embedding value")
plt.ylabel("Frequency")
plt.legend()

# ذخیره هیستوگرام
plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=300)
plt.close()

print("Plot saved:", OUT_PLOT)



#cd D:\PayanNameh\paper-structure\deep-keypoints-attack
#.\venv\Scripts\activate
#python tools\defense_evaluate.py

