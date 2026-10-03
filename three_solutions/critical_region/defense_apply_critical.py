import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

def get_embedding(img):
    faces = app.get(img)
    if len(faces) == 0:
        return None
    return faces[0].embedding

def defend_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    
    # ناحیه بحرانی: چشم‌ها، بینی، دهان
    h, w = img.shape[:2]
    
    mask = np.zeros((h, w), dtype=np.uint8)

    # سه دایره روی ناحیه چشم‌ها و بینی: دفاع سبک ولی موثر
    cv2.circle(mask, (int(w*0.35), int(h*0.38)), 40, 255, -1)
    cv2.circle(mask, (int(w*0.65), int(h*0.38)), 40, 255, -1)
    cv2.circle(mask, (int(w*0.50), int(h*0.55)), 50, 255, -1)

    blurred = cv2.GaussianBlur(img, (31, 31), 0)
    
    defended = img.copy()
    defended[mask == 255] = blurred[mask == 255]

    return get_embedding(defended)
