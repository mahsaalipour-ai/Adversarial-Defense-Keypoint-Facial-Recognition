#extract_defense_embeddings
import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEF_DIR = os.path.join(BASE, "data", "defense_out")
SAMPLE_OUT_DIR = os.path.join(BASE, "tools", "extract_defense_embeddings")
os.makedirs(SAMPLE_OUT_DIR, exist_ok=True)

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

SAMPLE_ID = "000001" 
def_img_path = os.path.join(DEF_DIR, f"{SAMPLE_ID}_def.png") # نام فایل دفاع شده شما

if not os.path.exists(def_img_path):
    def_img_path = sorted(glob.glob(os.path.join(DEF_DIR, "*.png")))[0]
    SAMPLE_ID = os.path.basename(def_img_path).split("_")[0]

img = cv2.imread(def_img_path)
faces = app.get(img)
sample_emb = faces[0].embedding.astype(np.float32)

# --- تولید Heatmap بردار ویژگی (دفاع) ---
plt.figure(figsize=(15, 1.5))
plt.imshow(sample_emb.reshape(1, -1), aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
plt.colorbar(label="Val")
plt.yticks([])
plt.title(f"Defense Embedding Heatmap (ID={SAMPLE_ID})")
plt.tight_layout()
plt.savefig(os.path.join(SAMPLE_OUT_DIR, f"{SAMPLE_ID}_defense_heatmap.png"), dpi=300)
plt.close()

print(f"Defense Heatmap saved for ID {SAMPLE_ID}")




#هر سه تا خط زیر رو باید با هم بزنی
#cd D:\PayanNameh\paper-structure\deep-keypoints-attack
#python tools\extract_defense_embeddings.py


