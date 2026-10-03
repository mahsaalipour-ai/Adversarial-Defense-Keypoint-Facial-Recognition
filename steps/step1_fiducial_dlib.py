# step1_fiducial_dlib.py
import dlib, cv2
import numpy as np
from pathlib import Path

# مسیر فایل مدل dlib و عکس صورت
base_dir = Path(__file__).parent.parent
predictor_path = base_dir / "models" / "shape_predictor_68_face_landmarks.dat"
#image_path = base_dir / "steps" / "assets" / "face1.jpg"
image_path = base_dir / "data" / "images" / "000001.jpg"

# بارگذاری مدل و تصویر
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(predictor_path))
image = cv2.imread(str(image_path))
assert image is not None, f"Image not found at {image_path}"
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# تشخیص چهره
faces = detector(gray)
if len(faces) == 0:
    print("❌ No face detected.")
    exit()

# استخراج ۶۸ نقطه
landmarks = predictor(gray, faces[0])
points = np.array([[p.x, p.y] for p in landmarks.parts()], dtype=np.float32)

# مسیر ذخیره در data/
out_path = base_dir / "data" / "face_landmarks_dlib.npy"
np.save(str(out_path), points)
print("✅Saved:", out_path)


#for run
#python step1_fiducial_dlib.py
