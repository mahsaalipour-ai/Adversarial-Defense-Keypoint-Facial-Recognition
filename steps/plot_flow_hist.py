# plot_flow_hist.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

base = Path(__file__).parent.parent
data = base / "data"

# اول از CSV بخونیم؛ اگه نبود از NPY
csv_path = data / "step2_points_table.csv"
if csv_path.exists():
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    flow = arr[:, 4:6]  # dx,dy
else:
    flow = np.load(data / "flow_step2.npy")

norms = np.linalg.norm(flow, axis=1)

print(f"points: {len(norms)}, moved: {(norms>0).sum()}")

# هیستوگرام
plt.figure(figsize=(7,4))
plt.hist(norms, bins=40)
plt.xlabel("|flow| (pixels)")
plt.ylabel("count")
plt.title("Histogram of displacement magnitudes")
plt.tight_layout()

# ذخیره
out_png = data / "flow_hist.png"
plt.savefig(out_png, dpi=150)
print("saved:", out_png)

# نمایش
plt.show()


#for run: python plot_flow_hist.py
#and you need matplotlib, if you don't have it: pip install matplotlib