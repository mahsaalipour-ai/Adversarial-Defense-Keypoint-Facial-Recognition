# step1_merge_all.py
import numpy as np
from pathlib import Path

base_dir = Path(__file__).parent.parent
data_dir = base_dir / "data"

# بارگذاری نقاط
dlib_pts = np.load(data_dir / "face_landmarks_dlib.npy")  # حدود 68 نقطه
edge_pts = np.load(data_dir / "face_edges.npy")           # هزاران نقطه
mser_pts = np.load(data_dir / "face_mser.npy")            # چند هزار نقطه

# ادغام
all_points = np.concatenate([dlib_pts, edge_pts, mser_pts], axis=0)

# حذف نقاط تکراری
uniq = np.unique(all_points, axis=0)

# انتخاب زیرمجموعه ۲٪ پیکسلی (مثلاً 224×224)
max_points = int(0.02 * 224 * 224)
selected = uniq[:max_points] if len(uniq) > max_points else uniq

# ذخیره نهایی در data/
out_path = data_dir / "final_keypoints.npy"
np.save(str(out_path), selected)
print(f"✅ Saved: {out_path} with {len(selected)} points.")
