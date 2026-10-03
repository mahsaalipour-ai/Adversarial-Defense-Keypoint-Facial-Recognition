# tools/defense_apply.py
import cv2
import numpy as np
from pathlib import Path

#def squeeze(img):
    # بهترین فیلتر دفاعی ترکیبی
    #sq = cv2.fastNlMeansDenoisingColored(img, None, 8, 8, 7, 21)
    #return sq

def squeeze(img):
    return cv2.GaussianBlur(img, (5,5), 1.2)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    img = cv2.imread(args.img)
    if img is None:
        raise ValueError("Could not read image")

    defended = squeeze(img)
    cv2.imwrite(args.out, defended)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
