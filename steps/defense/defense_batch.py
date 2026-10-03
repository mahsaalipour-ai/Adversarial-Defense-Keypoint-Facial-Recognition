# steps/defense/defense_batch.py

import cv2
from pathlib import Path
import sys

# ===========================
# مسیر درست tools برای import
# ===========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "tools"
sys.path.append(str(TOOLS_DIR))

from defense_apply import squeeze

# ===========================
# مسیرهای اصلی
# ===========================
STEP3_DIR = PROJECT_ROOT / "data" / "step3"
DEF_DIR   = PROJECT_ROOT / "data" / "defense_out"

DEF_DIR.mkdir(parents=True, exist_ok=True)

# ===========================
# پیدا کردن تصاویر حمله
# ===========================
imgs = sorted(STEP3_DIR.glob("*_adv_step3.png"))
print(f"[info] found {len(imgs)} images in {STEP3_DIR}")

done = 0

# ===========================
# پردازش batch
# ===========================
for img_path in imgs:
    base_id = img_path.stem.split("_")[0]  # مثلاً 000001
    out_path = DEF_DIR / f"{base_id}_def.png"

    img = cv2.imread(str(img_path))
    if img is None:
        print("[warn] cannot read:", img_path)
        continue

    defended = squeeze(img)
    cv2.imwrite(str(out_path), defended)

    done += 1
    print(f"[ok] {img_path.name} -> {out_path.name}")

print(f"\n✅ finished. defended images: {done}")
print(f"output folder: {DEF_DIR}")


#برای اجرا هر سه خط زیر رو با هم بزن
#cd D:\PayanNameh\paper-structure\deep-keypoints-attack
#.\venv\Scripts\activate
#python steps\defense\defense_batch.py

