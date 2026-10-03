# inspect_dlib_points.py
import cv2
import numpy as np
from pathlib import Path

base_dir = Path(__file__).parent.parent
image_path = base_dir / "data" / "images" / "000001.jpg"   # یکی از عکس‌های دیتاست
pts_path   = base_dir / "data" / "face_landmarks_dlib.npy"

# بارگذاری
image = cv2.imread(str(image_path))
pts   = np.load(str(pts_path))

# رسم نقاط
for (x, y) in pts.astype(int):
    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

cv2.imshow("DLIB landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#for run
#python inspect_dlib_points.py