# env_defense.ps1 — defense demo params (minimal & reproducible)

# paths
$env:IMAGE_PATH = "D:\PayanNameh\paper-structure\deep-keypoints-attack\data\images\000001.jpg"
$env:STEP2_OUT_DIR = "D:\PayanNameh\paper-structure\deep-keypoints-attack\data\step2"
$env:STEP3_OUT_DIR = "D:\PayanNameh\paper-structure\deep-keypoints-attack\data\step3_out_def"

# feature-squeeze params
$env:SQUEEZE_METHOD = "pca"    # options: pca | quant
$env:SQUEEZE_PCA_DIM = "32"
$env:SQUEEZE_QUANT_BITS = "4"

# region-select params
$env:REGION_THRESHOLD = "0.2"   # threshold on normalized flow magnitude
$env:REGION_MIN_AREA = "200"    # min pixels for region

# adversarial-train (minimal)
$env:DEF_EPOCHS = "10"
$env:DEF_BATCH = "8"
$env:DEF_LR = "1e-4"

# misc
$env:HEADLESS = "1"
