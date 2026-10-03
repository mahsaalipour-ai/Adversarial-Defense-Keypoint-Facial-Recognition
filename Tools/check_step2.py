import numpy as np, pathlib
s2 = pathlib.Path("data/step2")
name = "000001"
dstf = s2 / f"{name}_dst_pts_step2.npy"
movedf = s2 / f"{name}_moved_idx_step2.npy"
print("dst exists:", dstf.exists(), "moved exists:", movedf.exists())
if dstf.exists():
    pts = np.load(dstf)
    print("dst shape, min, max:", pts.shape, pts.min(), pts.max())
if movedf.exists():
    mi = np.load(movedf)
    print("moved_idx shape, sample(10):", mi.shape, mi[:10])
