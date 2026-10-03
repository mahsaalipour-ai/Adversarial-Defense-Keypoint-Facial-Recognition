import os
import numpy as np
from defense_apply_critical import defend_image

INPUT_DIR = r"D:\PayanNameh\paper-structure\deep-keypoints-attack\data\step3"
OUT_PATH = "results_critical.npy"

embs = {}
files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".png") or f.endswith(".jpg")])

for f in files:
    emb = defend_image(os.path.join(INPUT_DIR, f))
    if emb is not None:
        embs[f] = emb

np.save(OUT_PATH, embs)
print("DONE. Saved", OUT_PATH)
