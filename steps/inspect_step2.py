# inspect_step2.py
# inspect_step2.py (CSV-based version)
import numpy as np
from pathlib import Path

data_dir = Path(__file__).parent.parent / "data"
csv = data_dir / "step2_points_table.csv"
arr = np.loadtxt(csv, delimiter=",", skiprows=1)

# --- استخراج ستون‌ها از CSV ---
src  = arr[:,0:2].astype(np.float32)
dst  = arr[:,2:4].astype(np.float32)
flow = arr[:,4:6].astype(np.float32)

# --- تشخیص نقاط جابه‌جا‌شده بر اساس آستانه ---
moved_mask = np.linalg.norm(flow, axis=1) > 1e-6
moved = np.where(moved_mask)[0].astype(np.int32)

# --- چاپ اطلاعات ---
print("shapes:")
print(" src:", src.shape, " dtype:", src.dtype)
print(" dst:", dst.shape, " dtype:", dst.dtype)
print(" flow:", flow.shape, " dtype:", flow.dtype)
print(" moved_idx:", moved.shape, " dtype:", moved.dtype)
print()

# آمار کلی
print("flow norms (min/mean/max):")
norms = (flow**2).sum(axis=1)**0.5
print(" min:", norms.min(), " mean:", norms.mean(), " max:", norms.max())
print()

# چند نمونه نخست هر آرایه
n_show = 8
print("first", n_show, "src:")
print(src[:n_show])
print("first", n_show, "dst:")
print(dst[:n_show])
print("first", n_show, "flow:")
print(flow[:n_show])
print()
print("moved indices (first 50):", moved[:50])


#for run:
#python inspect_step2.py
