# check_pts_auto.py
import glob, os, numpy as np

candidates = glob.glob(os.path.join("data", "step2", "*_src_pts_step2.npy"))
if not candidates:
    candidates = glob.glob(os.path.join("**", "*_src_pts_step2.npy"), recursive=True)

if not candidates:
    print("❗ هیچ فایل *_src_pts_step2.npy پیدا نشد. اجرا کن step2_warp.py تا ساخته بشه.")
    raise SystemExit(1)

for src_path in candidates:
    base = os.path.basename(src_path).replace("_src_pts_step2.npy","")
    d = os.path.dirname(src_path)
    src = np.load(src_path)
    dst = np.load(os.path.join(d, f"{base}_dst_pts_step2.npy"))
    moved = np.load(os.path.join(d, f"{base}_moved_idx_step2.npy"))
    print(f"\n=== {base} ===")
    print("SRC shape:", src.shape)
    print("SRC x min,max:", float(src[:,0].min()), float(src[:,0].max()))
    print("SRC y min,max:", float(src[:,1].min()), float(src[:,1].max()))
    print("---")
    print("DST shape:", dst.shape)
    print("DST x min,max:", float(dst[:,0].min()), float(dst[:,0].max()))
    print("DST y min,max:", float(dst[:,1].min()), float(dst[:,1].max()))
    print("---")
    print("moved sample (first 20):", moved[:20])
    if dst.max() <= 1.01:
        print(">> NOTE: dst appears normalized in [0..1]. NEED to scale by image width/height.")
    else:
        print(">> NOTE: dst appears to be in pixel coords (good).")
