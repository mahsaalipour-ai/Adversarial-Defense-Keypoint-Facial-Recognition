import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from defense_apply_compress import bit_depth_squeeze, get_embedding

# -----------------------------
# تنظیمات مسیرها
# -----------------------------
BASE_OUT_DIR = r"D:\PayanNameh\paper-structure\deep-keypoints-attack\three_solutions\feature_compress"
IMG_PATH = r"D:\PayanNameh\paper-structure\deep-keypoints-attack\data\step3\000001_adv_step3.png"

# اطمینان از وجود پوشه خروجی
os.makedirs(BASE_OUT_DIR, exist_ok=True)

# -----------------------------
# ۱. بارگذاری تصویر متخاصم
# -----------------------------
img_adv = cv2.imread(IMG_PATH)
if img_adv is None:
    raise FileNotFoundError("Image not found! Check IMG_PATH.")

# -----------------------------
# ۲. استخراج embedding قبل از فشرده‌سازی
# -----------------------------
emb_before = get_embedding(img_adv)

# -----------------------------
# ۳. اعمال فشرده‌سازی
# -----------------------------
img_squeezed = bit_depth_squeeze(img_adv, bits=4)
emb_after = get_embedding(img_squeezed)

# -----------------------------
# ۴. رسم نمودار embedding
# -----------------------------
plt.figure(figsize=(12, 5))
x = np.arange(50)

plt.plot(x, emb_before[:50], label="Adversarial (Before Squeezing)",
         linestyle="--", alpha=0.7)
plt.plot(x, emb_after[:50], label="Defended (After 4-bit Squeezing)",
         linewidth=2)

plt.title("Effect of Feature Squeezing on Embedding Vector (First 50 Dimensions)")
plt.xlabel("Feature Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True, linestyle=":")

plot_path = os.path.join(BASE_OUT_DIR, "feature_squeezing_comparison.png")
plt.savefig(plot_path)
plt.close()  # بسته کردن نمودار بعد از ذخیره آن

# -----------------------------
# ۵. رسم هیت‌مپ مقایسه‌ای (برای ۵۰ درایه اول)
# -----------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

heatmap_before = emb_before[:50].reshape(1, -1)
heatmap_after = emb_after[:50].reshape(1, -1)

# هیت‌مپ قبل
im1 = ax1.imshow(heatmap_before, cmap='RdYlBu', aspect='auto')
ax1.set_title("Feature Heatmap: Stage 1 Output (Before Squeezing)")
ax1.set_yticks([])

# هیت‌مپ بعد
im2 = ax2.imshow(heatmap_after, cmap='RdYlBu', aspect='auto')
ax2.set_title("Feature Heatmap: Stage 2 Output (After 4-bit Squeezing)")
ax2.set_yticks([])

# افزودن Colorbar مشترک
fig.colorbar(im2, ax=[ax1, ax2], orientation='horizontal', pad=0.15, label='Intensity')

heatmap_path = os.path.join(BASE_OUT_DIR, "heatmap_comparison.png")
plt.savefig(heatmap_path)
plt.close()  # بسته کردن هیت‌مپ بعد از ذخیره آن

# -----------------------------
# ۶. ذخیره تصویر فشرده‌شده
# -----------------------------
img_out_path = os.path.join(BASE_OUT_DIR, "compressed_face_sample.png")
cv2.imwrite(img_out_path, img_squeezed)

# -----------------------------
# ۷. نمایش فایل‌های ذخیره‌شده
# -----------------------------
print("Saved files:")
print(f"Plot: {plot_path}")
print(f"Heatmap: {heatmap_path}")
print(f"Compressed Image: {img_out_path}")




# for run:
#cd D:\PayanNameh\paper-structure\deep-keypoints-attack
#.\venv\Scripts\activate
#python three_solutions\feature_compress\visualize_squeezing.py
