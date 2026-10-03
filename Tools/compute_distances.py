import numpy as np
import os
import csv
import matplotlib.pyplot as plt

# -----------------------------
# مسیر فایل‌ها
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DIR = os.path.join(BASE_DIR, "data", "embeddings")

clean_emb = np.load(os.path.join(DIR, "clean_arcface.npy"))
attack_emb = np.load(os.path.join(DIR, "attack_arcface.npy"))
defense_emb = np.load(os.path.join(DIR, "defense_arcface.npy"))

# id ها
clean_ids = [i.strip() for i in open(os.path.join(DIR, "clean_arcface_ids.txt")).readlines()]
attack_ids = [i.strip() for i in open(os.path.join(DIR, "attack_arcface_ids.txt")).readlines()]
defense_ids = [i.strip() for i in open(os.path.join(DIR, "defense_arcface_ids.txt")).readlines()]

# -----------------------------
# تابع فاصله L2
# -----------------------------
def dist(a, b):
    return np.linalg.norm(a - b)

# -----------------------------
# محاسبه فاصله‌ها
# -----------------------------
results = []

dist_ca = []  # clean vs attack
dist_cd = []  # clean vs defense
dist_ad = []  # attack vs defense

for atk_id in attack_ids:

    if atk_id not in clean_ids or atk_id not in defense_ids:
        continue

    clean_idx = clean_ids.index(atk_id)
    attack_idx = attack_ids.index(atk_id)
    defense_idx = defense_ids.index(atk_id)

    d_ca = dist(clean_emb[clean_idx], attack_emb[attack_idx])
    d_cd = dist(clean_emb[clean_idx], defense_emb[defense_idx])
    d_ad = dist(attack_emb[attack_idx], defense_emb[defense_idx])

    results.append([atk_id, d_ca, d_cd, d_ad])

    dist_ca.append(d_ca)
    dist_cd.append(d_cd)
    dist_ad.append(d_ad)

# -----------------------------
# ذخیره CSV
# -----------------------------
out_path = os.path.join(DIR, "distances_arcface.csv")

with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "id",
        "dist_clean_attack",
        "dist_clean_defense",
        "dist_attack_defense"
    ])
    w.writerows(results)

print("Saved:", out_path)
print("Rows:", len(results))

# =====================================================
# رسم boxplot فاصله‌های کسینوسی
# =====================================================
plt.figure(figsize=(7, 5))

plt.boxplot(
    [dist_ca, dist_cd, dist_ad],
    labels=[
        "Clean – Attack",
        "Clean – Defense",
        "Attack – Defense"
    ],
    showfliers=True
)

plt.ylabel("L2 Distance (ArcFace Embedding)")
plt.title("Distribution of Embedding Distances (ArcFace)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plot_path = os.path.join(DIR, "boxplot_distances_arcface.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print("Boxplot saved to:", plot_path)
print("Done.")



#هر سه تای خط های زیر رو زیر هم بزن
#cd D:\PayanNameh\paper-structure\deep-keypoints-attack
#.\venv\Scripts\activate
#python tools\compute_distances.py
