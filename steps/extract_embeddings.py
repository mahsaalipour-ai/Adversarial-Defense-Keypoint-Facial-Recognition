import os
import cv2
import numpy as np
import onnxruntime as ort

# ---------- مسیرها ----------
MODEL_PATH = r"D:/PayanNameh/paper-structure/deep-keypoints-attack/models/arcface.onnx"

IMAGE_BEFORE = r"D:/PayanNameh/paper-structure/deep-keypoints-attack/data/images/000001.jpg"
IMAGE_AFTER = r"D:/PayanNameh/paper-structure/deep-keypoints-attack/data/step3/000001_adv_step3.png"

OUT_BEFORE = r"D:/PayanNameh/paper-structure/deep-keypoints-attack/steps/embeddings_before_attack.npy"
OUT_AFTER = r"D:/PayanNameh/paper-structure/deep-keypoints-attack/steps/embeddings_after_attack.npy"


# ---------- بارگذاری مدل ----------
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name


# ---------- پیش‌پردازش ----------
def preprocess_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"تصویر پیدا نشد: {path}")

    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"خواندن تصویر ناموفق بود: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))   # (C, H, W)
    img = np.expand_dims(img, axis=0)    # (1, C, H, W)
    return img


# ---------- استخراج embedding ----------
def extract_embedding(image_path):
    img = preprocess_image(image_path)
    emb = session.run(None, {input_name: img})[0]
    emb = emb.squeeze()                  # (D,)
    emb = emb / np.linalg.norm(emb)      # نرمال‌سازی
    return emb


# ---------- اجرا ----------
if __name__ == "__main__":
    emb_before = extract_embedding(IMAGE_BEFORE)
    emb_after = extract_embedding(IMAGE_AFTER)

    np.save(OUT_BEFORE, emb_before)
    np.save(OUT_AFTER, emb_after)

    print("Embedding before_attack:", emb_before.shape)
    print("Embedding after_attack:", emb_after.shape)
    print("All files are saved.")


# برای اجرا هر دو تا پایینیا رو با هم بزن
# cd D:\PayanNameh\paper-structure\deep-keypoints-attack
# python steps\extract_embeddings.py
