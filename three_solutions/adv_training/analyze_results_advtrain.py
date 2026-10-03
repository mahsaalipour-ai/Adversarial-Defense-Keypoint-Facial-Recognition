import numpy as np
import os
import matplotlib.pyplot as plt

# مسیر ذخیره خروجی
OUT_DIR = r"D:\PayanNameh\paper-structure\deep-keypoints-attack\three_solutions\adv_training"

# مسیر embeddingهای اصلی
EMB_DIR = r"D:\PayanNameh\paper-structure\deep-keypoints-attack\data\embeddings"

clean = np.load(os.path.join(EMB_DIR, "clean_arcface.npy"), allow_pickle=True)
attack = np.load(os.path.join(EMB_DIR, "attack_arcface.npy"), allow_pickle=True)

# نتایج دفاع آموزش تقابلی
defense = np.load(os.path.join(OUT_DIR, "results_advtrain.npy"), allow_pickle=True).item()

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

correct_def = 0
dist_clean_def = []
dist_attack_def = []

valid_sample_found = False

for fname, emb_def in defense.items():
    idx = int(fname.split("_")[0])

    if idx >= len(clean):
        continue

    emb_clean = clean[idx]
    emb_attack = attack[idx]

    # ذخیره یک نمونه برای هیستوگرام
    if not valid_sample_found:
        sample_clean = emb_clean
        sample_attack = emb_attack
        sample_def = emb_def
        valid_sample_found = True

    # دقت دفاع
    if cosine(emb_def, emb_clean) > 0.5:
        correct_def += 1

    # فاصله‌ها
    dist_clean_def.append(np.linalg.norm(emb_clean - emb_def))
    dist_attack_def.append(np.linalg.norm(emb_attack - emb_def))

# نتایج نهایی
acc = correct_def / len(defense)
d_clean = np.mean(dist_clean_def)
d_attack = np.mean(dist_attack_def)

print("Defense Accuracy =", acc)
print("dist_clean_def =", d_clean)
print("dist_attack_def =", d_attack)

# --------------------------
#     ذخیره فایل TXT
# --------------------------

txt_path = os.path.join(OUT_DIR, "advtrain_results.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("Defense Method: Adversarial Training\n")
    f.write(f"Defense Accuracy: {acc}\n")
    f.write(f"dist_clean_def: {d_clean}\n")
    f.write(f"dist_attack_def: {d_attack}\n")
    f.write("\n(Generated Automatically)\n")

print("TXT saved to:", txt_path)

# --------------------------
#     هیستوگرام‌ها
# --------------------------

plt.figure(figsize=(10, 5))
plt.title("Clean Embedding Histogram")
plt.hist(sample_clean, bins=50, color='green')
plt.savefig(os.path.join(OUT_DIR, "hist_clean.png"))
plt.close()

plt.figure(figsize=(10, 5))
plt.title("Attack Embedding Histogram")
plt.hist(sample_attack, bins=50, color='red')
plt.savefig(os.path.join(OUT_DIR, "hist_attack.png"))
plt.close()

plt.figure(figsize=(10, 5))
plt.title("Defense (Adv Training) Embedding Histogram")
plt.hist(sample_def, bins=50, color='blue')
plt.savefig(os.path.join(OUT_DIR, "hist_defense_advtrain.png"))
plt.close()

print("Histograms saved.")


#for run: python analyze_results_advtrain.py
