import cv2
import numpy as np
from insightface.app import FaceAnalysis

# مدل ArcFace
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

def add_adv_training_noise(img, eps=2.0):
    """
    دفاع آموزش تقابلی ساده:
    یک نویز کوچک کنترل‌شده اضافه می‌کنیم
    تا embedding نسبت به تغییرات حمله پایدارتر شود.
    """
    noise = np.random.normal(0, eps, img.shape).astype(np.float32)
    adv_img = img.astype(np.float32) + noise
    adv_img = np.clip(adv_img, 0, 255).astype(np.uint8)
    return adv_img


def get_embedding(img):
    faces = app.get(img)
    if len(faces) == 0:
        return None
    return faces[0].embedding


def defend_image(path_in):
    img = cv2.imread(path_in)
    if img is None:
        return None

    # دفاع مبتنی بر آموزش تقابلی
    defended = add_adv_training_noise(img)

    emb = get_embedding(defended)
    return emb
