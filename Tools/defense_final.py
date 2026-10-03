# tools/defense_final.py
import cv2
import numpy as np
from pathlib import Path
import argparse

# -------------------
#  utils
# -------------------
def mse(a, b):
    return np.mean((a.astype(float) - b.astype(float)) ** 2)


# 1) bit depth squeeze
def squeeze_bit(img, bits):
    step = 256 // (2 ** bits)
    return (img // step) * step


# 2) median filter
def squeeze_median(img, k):
    return cv2.medianBlur(img, k)


# 3) bilateral
def squeeze_bilateral(img, d, sc, ss):
    return cv2.bilateralFilter(img, d, sc, ss)


# 4) NLM (slow but best)
def squeeze_nlm(img, h=8, tmpl=7, search=21):
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, tmpl, search)


# ---------------------------------------------------
# main defense pipeline
# ---------------------------------------------------
def run_defense(img_path, tau):
    img_path = Path(img_path)
    name = img_path.stem.replace("_region", "")
    out_report = img_path.parent / f"{name}_final_report.txt"

    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to read {img_path}")

    divergences = []

    # bit depth
    for b in [5, 4]:
        sq = squeeze_bit(img, b)
        divergences.append(mse(img, sq))

    # median
    for k in [3, 5]:
        sq = squeeze_median(img, k)
        divergences.append(mse(img, sq))

    # bilateral
    sq = squeeze_bilateral(img, 5, 40, 7)
    divergences.append(mse(img, sq))

    # NLM
    sq = squeeze_nlm(img, h=8, tmpl=7, search=21)
    divergences.append(mse(img, sq))

    max_div = max(divergences)
    flagged = int(max_div > tau)

    # save report
    with open(out_report, "w") as f:
        f.write(f"image: {name}\n")
        f.write(f"tau: {tau}\n")
        f.write("divergences:\n")
        for d in divergences:
            f.write(f"  - {d:.5f}\n")
        f.write(f"max_div: {max_div:.5f}\n")
        f.write(f"flagged_adv: {flagged}\n")

    return flagged, max_div, out_report


# ---------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="path to *_region.png")
    ap.add_argument("--tau", type=float, required=True)
    args = ap.parse_args()

    flagged, max_div, rep = run_defense(args.img, args.tau)
    print("done:", flagged, max_div, rep)


if __name__ == "__main__":
    main()
