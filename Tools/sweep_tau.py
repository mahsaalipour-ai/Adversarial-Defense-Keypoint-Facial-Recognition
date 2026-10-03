import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(__file__))

CSV_PATH = os.path.join(BASE, "data", "embeddings", "distances_arcface.csv")
DIV_PATH = os.path.join(BASE, "data", "defense_out", "divergence_all.npy")

# فایلی که در defense_batch ذخیره شده
divs = np.load(DIV_PATH)   # طول = 100

df = pd.read_csv(CSV_PATH)

# حمله موفق یعنی: dist_clean_attack > dist_clean_defense
df["attack_wrong"] = df["dist_clean_attack"] > df["dist_clean_defense"]

attack_mask = df["attack_wrong"].values.astype(int)

taus = np.linspace(5, 30, 100)

best_tau = None
best_dsr = -1

results = []

for tau in taus:
    detected = divs > tau      # 1 یعنی حمله تشخیص داده شده
    dsr = np.mean(detected & attack_mask)   # فقط حمله‌های واقعی

    results.append((tau, dsr))

    if dsr > best_dsr:
        best_dsr = dsr
        best_tau = tau

print("\n===== SWEEP RESULT =====")
print("Best tau:", best_tau)
print("Best Defense Success Rate (DSR):", best_dsr)

# ذخیره نتیجه کامل sweep برای رسم نمودار
OUT = os.path.join(BASE, "data", "embeddings", "tau_sweep_results.csv")
import csv
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tau", "DSR"])
    w.writerows(results)

print("\nSaved:", OUT)

# برای اجرا هر سه خط پایین رو با هم بزن
#cd D:\PayanNameh\paper-structure\deep-keypoints-attack
#.\venv\Scripts\activate
#python tools\sweep_tau.py
