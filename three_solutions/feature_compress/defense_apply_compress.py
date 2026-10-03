import cv2
import numpy as np
from insightface.app import FaceAnalysis

# مدل ArcFace
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

def bit_depth_squeeze(img, bits=4):
    """
    فشرده‌سازی ویژگی با کاهش تعداد بیت رنگ.
    """
    shift = 8 - bits       # مثلاً 8-4 = 4
    squeezed = (img >> shift) << shift
    return squeezed

def get_embedding(img):
    faces = app.get(img)
    if len(faces) == 0:
        return None
    return faces[0].embedding

def defend_image(path):
    img = cv2.imread(path)
    if img is None:
        return None

    squeezed = bit_depth_squeeze(img, bits=4)
    emb = get_embedding(squeezed)
    return emb
