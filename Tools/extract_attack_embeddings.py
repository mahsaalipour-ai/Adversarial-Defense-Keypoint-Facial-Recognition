#extract_attack_embeddings
import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEP3_DIR = os.path.join(BASE_DIR, "data", "step3")
SAMPLE_OUT_DIR = os.path.join(BASE_DIR, "tools", "extract_attack_embeddings")
os.makedirs(SAMPLE_OUT_DIR, exist_ok=True)

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

SAMPLE_ID = "000001" 
attack_img_path = os.path.join(STEP3_DIR, f"{SAMPLE_ID}_adv_step3.png")

if not os.path.exists(attack_img_path):
    attack_img_path = sorted(glob.glob(os.path.join(STEP3_DIR, "*_adv_step3.png")))[0]
    SAMPLE_ID = os.path.basename(attack_img_path).split("_")[0]

img = cv2.imread(attack_img_path)
faces = app.get(img)
sample_emb = faces[0].embedding.astype(np.float32)

# --- تولید Heatmap بردار ویژگی (حمله) ---
plt.figure(figsize=(15, 1.5))
plt.imshow(sample_emb.reshape(1, -1), aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
plt.colorbar(label="Val")
plt.yticks([])
plt.title(f"Attack Embedding Heatmap (ID={SAMPLE_ID})")
plt.tight_layout()
plt.savefig(os.path.join(SAMPLE_OUT_DIR, f"{SAMPLE_ID}_attack_heatmap.png"), dpi=300)
plt.close()

print(f"Attack Heatmap saved for ID {SAMPLE_ID}")



#برای اجرا هر سه تا رو پشت هم بزن و با هم
#cd D:\PayanNameh\paper-structure\deep-keypoints-attack
#.\venv\Scripts\activate
#python tools\extract_attack_embeddings.py

