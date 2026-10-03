# step1_edges.py
import cv2
import numpy as np
from skimage.filters import prewitt
from pathlib import Path

# مسیر به فایل تصویر (face.jpg) داخل assets
img_path = Path(__file__).parent / "assets" / "face1.jpg"
img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
assert img is not None, f"Image not found at {img_path}"

# لبه‌یابی با Canny
edges_canny = cv2.Canny(img, 100, 200)

# لبه‌یابی با Prewitt
edges_prewitt = (prewitt(img) * 255).astype(np.uint8)

# ترکیب دو لبه
edges_combined = cv2.bitwise_or(edges_canny, edges_prewitt)

# استخراج مختصات نقاط لبه
ys, xs = np.where(edges_combined > 0)
edge_points = np.stack((xs, ys), axis=1).astype(np.float32)

# مسیر ذخیره خروجی داخل پوشه data
out_path = Path(__file__).parent.parent / "data" / "face_edges.npy"
np.save(str(out_path), edge_points)
print("✅Saved:", out_path)
