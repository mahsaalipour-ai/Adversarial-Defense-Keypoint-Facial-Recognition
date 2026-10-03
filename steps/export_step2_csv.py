# export_step2_csv.py
import numpy as np
from pathlib import Path

# مسیر پوشه‌ی پروژه (فایل رو داخل فولدر steps گذاشتی یا بالاتر؟)
# این فرض رو می‌گیریم که فایل را داخل فولدر steps اجرا می‌کنی؛ این مسیر داده‌ها رو به درستی تنظیم می‌کنه.
# اگر این فایل را در steps قرار دادی، parent.parent -> project root
base = Path(__file__).parent.parent
data_dir = base / "data"

# بارگذاری فایل‌های تولیدشده در step2
src = np.load(data_dir / "src_pts_step2.npy")
dst = np.load(data_dir / "dst_pts_step2.npy")
flow = np.load(data_dir / "flow_step2.npy")

# ترکیب ستون‌ها: x_src,y_src,x_dst,y_dst,dx,dy
arr = np.hstack([src, dst, flow])

out_csv = data_dir / "step2_points_table.csv"
np.savetxt(out_csv, arr, delimiter=",",
           header="x_src,y_src,x_dst,y_dst,dx,dy", comments="", fmt="%.6f")
print("✅ Saved CSV:", out_csv.resolve())

# for run: python export_step2_csv.py
