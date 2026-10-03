# env.ps1 — reproducible params (for step2 -> step3)
# تنظیمات تصویر / ورودی
$env:IMAGE_PATH = "D:\PayanNameh\paper-structure\deep-keypoints-attack\data\images\000001.jpg"

# STEP2 (geom perturbation)
$env:STEP2_PCT_GEOM = "0.25"     # درصد نقاطی که جابجا می‌شوند (0.02 برای حالت مقاله — این نسخه تقویتی)
$env:STEP2_JITTER   = "12"       # پیکسلِ جابجایی (قوی‌تر)
$env:STEP2_FLOW_SIGMA = "1.2"    # sigma برای dense flow

# STEP3 (mask optimization)
# تنظیم پارامترهای بهینه برای حمله طبیعی‌تر
# env_paper.ps1
$env:STEP2_PCT_GEOM = "0.02"
$env:STEP2_JITTER   = "5"
$env:STEP2_FLOW_SIGMA = "1.0"

$env:USE_GAN   = "0"
$env:ITERS     = "3000"
$env:ETA_PUSH  = "8.0"
$env:LR_MASK   = "0.02"
$env:HEADLESS  = "1"



# خروجی اختصاصی (اختیاری)
$env:STEP3_OUT = "D:\PayanNameh\paper-structure\deep-keypoints-attack\data\step3_out_expl"
