import numpy as np
import os
import matplotlib.pyplot as plt

OUT_DIR = r"D:\PayanNameh\paper-structure\deep-keypoints-attack\three_solutions\feature_compress"
EMB_DIR = r"D:\PayanNameh\paper-structure\deep-keypoints-attack\data\embeddings"

clean = np.load(os.path.join(EMB_DIR, "clean_arcface.npy"), allow_pickle=True)
attack = np.load(os.path.join(EMB_DIR, "attack_arcface.npy"), allow_pickle=True)

defense = np.load(os.path.join(OUT_DIR, "results_compress.npy"), allow_pickle=True).item()

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

correct_def = 0
dist_clean_def = []
dist_attack_def = []

sample_clean = None

for fname, emb_def in defense.items():
    idx = int(fname.split("_")[0])
    if idx >= len(clean):
        continue

    emb_clean = clean[idx]
    emb_attack = attack[idx]

    if sample_clean is None:
        sample_clean = emb_clean
        sample_attack = emb_attack
        sample_def = emb_def

    if cosine(emb_def, emb_clean) > 0.5:
        correct_def += 1

    dist_clean_def.append(np.linalg.norm(emb_clean - emb_def))
    dist_attack_def.append(np.linalg.norm(emb_attack - emb_def))

acc = correct_def / len(defense)
d_clean = np.mean(dist_clean_def)
d_attack = np.mean(dist_attack_def)

txt_path = os.path.join(OUT_DIR, "compress_results.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("Defense Method: Feature Compression (Bit-depth Reduction)\n")
    f.write(f"Defense Accuracy: {acc}\n")
    f.write(f"dist_clean_def: {d_clean}\n")
    f.write(f"dist_attack_def: {d_attack}\n")
    f.write("\n(Generated Automatically)\n")

print("TXT saved to:", txt_path)

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
plt.title("Defense (Compression) Embedding Histogram")
plt.hist(sample_def, bins=50, color='blue')
plt.savefig(os.path.join(OUT_DIR, "hist_defense_compress.png"))
plt.close()

print("Histograms saved.")
