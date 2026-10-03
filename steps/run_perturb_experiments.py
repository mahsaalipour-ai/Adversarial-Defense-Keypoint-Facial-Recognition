# run_perturb_experiments.py
import os
import sys
import shutil
import numpy as np
from pathlib import Path
# فرض کنیم در همین فایل توی پروژه تابع perturb_keypoints قابل import هست
# یا از همون فایلی که تابع رو گذاشتی import کن
from step2_warp import perturb_keypoints
import cv2

IMAGE_PATH = os.environ.get("IMAGE_PATH") or (
    sys.argv[1] if len(sys.argv) > 1 else None)
OUT_BASE = "data/step2_experiments"

if not IMAGE_PATH:
    print("Usage: set IMAGE_PATH or pass image path")
    sys.exit(1)

# مسیر npy ایجاد شده توسط step2_dlib_keypoints.py
landmarks_npy = "data/step2_dlib/000081_dlib68_kps.npy"
item = np.load(landmarks_npy, allow_pickle=True).item() if isinstance(np.load(
    landmarks_npy, allow_pickle=True), np.ndarray) else np.load(landmarks_npy, allow_pickle=True)
kps = item['kps']  # shape (68,2)
img = cv2.imread(IMAGE_PATH)

pcts = [0.02, 0.05, 0.10]
n_runs_per_pct = 3  # برای seed=None سه بار اجرا کن، برای دیدن تنوع

os.makedirs(OUT_BASE, exist_ok=True)

for pct in pcts:
    for run_id in range(n_runs_per_pct):
        # use seed=None to randomize; if you want deterministic use seed=run_id
        new_kps, moved_idx = perturb_keypoints(
            kps, pct=pct, seed=None, max_jitter_px=8)
        folder = os.path.join(OUT_BASE, f"pct_{int(pct*100)}", f"run_{run_id}")
        os.makedirs(folder, exist_ok=True)
        # save npy with moved_idx meta
        out_item = {'img_path': IMAGE_PATH, 'kps': new_kps, 'meta': {
            'pct': pct, 'run': run_id, 'moved_idx': moved_idx}}
        base = Path(IMAGE_PATH).stem
        out_npy = os.path.join(folder, f"{base}_perturbed.npy")
        np.save(out_npy, out_item, allow_pickle=True)
        # write preview overlay (show moved points in red)
        vis = img.copy()
        for (x, y) in kps.astype(int):
            cv2.circle(vis, (int(x), int(y)), 2,
                       (0, 255, 0), -1)  # original green
        for i in moved_idx:
            x, y = new_kps[i].astype(int)
            cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)  # moved red
        out_vis = os.path.join(folder, f"{base}_preview.jpg")
        cv2.imwrite(out_vis, vis)
        # print summary
        print(
            f"pct={pct} run={run_id} moved_count={len(moved_idx)} moved_idx={moved_idx} saved->{folder}")

# for run
# python run_perturb_experiments.py
