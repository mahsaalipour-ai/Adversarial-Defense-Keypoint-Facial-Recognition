# steps/step1_fiducial_dlib_batch.py
import dlib, cv2, numpy as np
from pathlib import Path

base = Path(__file__).parent.parent
predictor_path = base / "models" / "shape_predictor_68_face_landmarks.dat"
images_dir = base / "data" / "images"
out_dir = base / "data" / "landmarks_dlib"
out_dir.mkdir(parents=True, exist_ok=True)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(predictor_path))

imgs = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
ok = fail = 0

print(f"[info] found {len(imgs)} images in {images_dir}")
for i, p in enumerate(imgs, 1):
    img = cv2.imread(str(p))
    if img is None:
        print(f"[warn] skip {p.name}: cannot read")
        fail += 1; continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        print(f"[warn] no face -> {p.name}")
        fail += 1; continue

    lm = predictor(gray, faces[0])
    pts = np.array([[pt.x, pt.y] for pt in lm.parts()], dtype=np.float32)

    np.save(out_dir / f"{p.stem}_dlib68.npy", pts)
    ok += 1
    if i % 10 == 0 or i == len(imgs):
        print(f"[{i}/{len(imgs)}] saved {p.stem}_dlib68.npy")

print(f"\n✅ done. success={ok}, fail={fail}, out_dir={out_dir}")

#for run
#python step1_fiducial_dlib_batch.py