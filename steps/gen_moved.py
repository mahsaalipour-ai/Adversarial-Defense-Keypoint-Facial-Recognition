# steps/gen_moved.py
import numpy as np
src = np.load("../data/step2/000001_src_pts_step2.npy")
k = max(3, int(len(src)*0.05))
moved = np.random.choice(len(src), k, replace=False).astype(np.int32)
np.save("../data/step2/000001_moved_idx_step2.npy", moved)
print("moved_idx saved:", moved, "k=", k)

# python gen_moved.py

