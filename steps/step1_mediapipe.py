# step1_mediapipe.py
import os, cv2, numpy as np
from pathlib import Path
import mediapipe as mp

base = Path(__file__).parent.parent
data_dir = base / "data"
images_dir = data_dir / "images"
out_dir = data_dir / "landmarks_mp"   # خروجی‌های 468-نقطه‌ای
out_dir.mkdir(parents=True, exist_ok=True)

# ورودی: یا از ENV (برای تک‌عکس)، یا پیش‌فرض
img_path = os.environ.get("IMAGE_PATH")
if img_path: 
    imgs = [Path(img_path)]
else:
    imgs = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    if not imgs:
        raise FileNotFoundError(f"no images in {images_dir}")

mp_mesh = mp.solutions.face_mesh
# refine_landmarks=True اگر نقاط اطراف چشم/لب دقیق‌تر می‌خواهی
with mp_mesh.FaceMesh(static_image_mode=True, refine_landmarks=False, max_num_faces=1) as mesh:
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None: 
            print(f"[warn] skip (cannot read): {p.name}"); continue
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = mesh.process(rgb)
        if not res.multi_face_landmarks:
            print(f"[warn] no face: {p.name}"); continue

        lm = res.multi_face_landmarks[0]
        pts = np.array([[pt.x * w, pt.y * h] for pt in lm.landmark], dtype=np.float32)  # (468,2)

        # ذخیره سراسری + per-image
        np.save(data_dir / "face_landmarks.npy", pts)                   # برای run تکی
        np.save(out_dir / f"{p.stem}_mp468.npy", pts)                   # برای batch
        print(f"[ok] {p.name}: saved -> {out_dir / (p.stem + '_mp468.npy')} | N={len(pts)}")

#for run :
#python step1_mediapipe.py