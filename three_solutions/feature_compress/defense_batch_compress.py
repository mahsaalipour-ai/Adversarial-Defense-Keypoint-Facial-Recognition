import os
import numpy as np
from defense_apply_compress import defend_image

INPUT_DIR = r"D:\PayanNameh\paper-structure\deep-keypoints-attack\data\step3"
OUTPUT = "results_compress.npy"

embeddings = {}
files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".png") or f.endswith(".jpg")])

for f in files:
    emb = defend_image(os.path.join(INPUT_DIR, f))
    if emb is not None:
        embeddings[f] = emb

np.save(OUTPUT, embeddings)
print("DONE. Saved:", OUTPUT)
