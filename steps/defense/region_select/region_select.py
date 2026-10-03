# steps/defense/region_select.py
# Usage:
# python -m steps.defense.region_select --name 000001 --outdir data/defense_out --pad 0.3
import argparse, os
from pathlib import Path
import numpy as np
import cv2

def load_step2(name, step2_dir):
    # try common filenames
    base = Path(step2_dir)
    candidates = [
        base / f"{name}_moved_idx_step2.npy",
        base / f"{name}_moved_idx.npy",
    ]
    for c in candidates:
        if c.exists():
            return np.load(str(c))
    return None

def load_pts(name, step2_dir, which="dst"):
    base = Path(step2_dir)
    cand = base / f"{name}_{which}_pts_step2.npy"
    if cand.exists():
        return np.load(str(cand))
    # fallback: try without suffix
    cand2 = base / f"{name}_{which}_pts.npy"
    if cand2.exists():
        return np.load(str(cand2))
    return None

def bbox_from_points(pts):
    xs = pts[:,0]
    ys = pts[:,1]
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return int(x0), int(y0), int(x1), int(y1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--imgdir", default="data/step3")
    p.add_argument("--step2dir", default="data/step2")
    p.add_argument("--outdir", default="data/defense_out")
    p.add_argument("--pad", type=float, default=0.3)
    args = p.parse_args()

    name = args.name
    img_path = Path(args.imgdir) / f"{name}_adv_step3.png"
    if not img_path.exists():
        # try other suffixes
        for ext in [".png", ".jpg", ".jpeg"]:
            cand = Path(args.imgdir) / f"{name}{ext}"
            if cand.exists():
                img_path = cand; break
    if not img_path.exists():
        raise FileNotFoundError(img_path)

    dst_pts = load_pts(name, args.step2dir, which="dst")
    moved_idx = load_step2(name, args.step2dir)
    if dst_pts is None:
        # fallback: full image bbox
        img = cv2.imread(str(img_path))
        h,w = img.shape[:2]
        x0,y0,x1,y1 = 0,0,w,h
    else:
        if moved_idx is None or len(moved_idx) < 5:
            # too few moved points → use all points
            pts = dst_pts
        else:
            pts = dst_pts[np.array(moved_idx).astype(int)]

        x0,y0,x1,y1 = bbox_from_points(pts)
        img = cv2.imread(str(img_path))
        h,w = img.shape[:2]
        # pad
        ww = x1-x0; hh = y1-y0
        padx = int(ww * args.pad); pady = int(hh * args.pad)
        x0 = max(0, x0 - padx); y0 = max(0, y0 - pady)
        x1 = min(w, x1 + padx); y1 = min(h, y1 + pady)

    crop = cv2.imread(str(img_path))[y0:y1, x0:x1]
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    crop_path = Path(args.outdir) / f"{name}_region.png"
    cv2.imwrite(str(crop_path), crop)
    # also save a mask image (same size as original) for debugging
    mask = np.zeros((h,w), dtype='uint8')
    mask[y0:y1, x0:x1] = 255
    mask_path = Path(args.outdir) / f"{name}_region_mask.png"
    cv2.imwrite(str(mask_path), mask)
    print("wrote:", crop_path, mask_path)

if __name__ == "__main__":
    main()

# python -m steps.defense.region_select.region_select --name 000001 --outdir data/defense_out --pad 0.35
