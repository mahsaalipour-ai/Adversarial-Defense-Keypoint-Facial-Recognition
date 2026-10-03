import numpy as np, glob

p = r"D:/PayanNameh/paper-structure/deep-keypoints-attack/data/step2"
f = sorted(glob.glob(p + "/*_src_pts_step2.npy"))[0]

src  = np.load(f)
dst  = np.load(f.replace("_src_pts_", "_dst_pts_"))
flow = np.load(f.replace("_src_pts_", "_flow_"))
midx = np.load(f.replace("_src_pts_", "_moved_idx_"))

print("src.shape, dtype:",  src.shape,  src.dtype)
print("dst.shape, dtype:",  dst.shape,  dst.dtype)
print("flow.shape, dtype:", flow.shape, flow.dtype)
print("moved_idx.shape, dtype:", midx.shape, midx.dtype)
print("flow nonzero count:", (np.linalg.norm(flow,axis=1)>1e-6).sum(), "moved_idx len:", len(midx))
print("any NaN/Inf:", np.isnan(src).any() or np.isinf(src).any() or
                     np.isnan(dst).any() or np.isinf(dst).any())


# python check_step2.py
