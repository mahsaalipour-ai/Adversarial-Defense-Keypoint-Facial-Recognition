# steps/defense/squeeze.py
# Feature Squeezing defense (test-time): multi-squeezer divergence on ArcFace embeddings.
# Usage (PowerShell):
#   $env:ARC_MODEL=""   # optional: explicit .onnx path; else InsightFace FaceAnalysis used
#   python -m steps.defense.squeeze --img D:\...\data\step3\000001_after.jpg --tau 0.08 --denoise-out D:\...\data\defense_out\000001_clean.jpg

import os, sys, math, argparse
from pathlib import Path
import cv2
import numpy as np

# ---------- Embedding backend ----------
class ArcEmbedder:
    def __init__(self, onnx_path: str | None = None, ctx_id: int = -1):
        self.use_insightface = False
        self.model = None
        self.size = (112, 112)

        if onnx_path:
            # direct onnxruntime (keeps your current pipeline consistent)
            import onnxruntime as ort
            self.ort_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self.use_insightface = False
            self.input_name = self.ort_sess.get_inputs()[0].name
        else:
            # fallback: InsightFace FaceAnalysis
            try:
                from insightface.app import FaceAnalysis
                self.app = FaceAnalysis(name="buffalo_l")
                self.app.prepare(ctx_id=ctx_id, det_size=(320, 320))
                self.use_insightface = True
            except Exception as e:
                raise RuntimeError("No ONNX model given and InsightFace not available.") from e

    @staticmethod
    def _preproc_arc(x_bgr: np.ndarray, size=(112,112)):
        x = cv2.cvtColor(x_bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, size)
        x = (x.astype(np.float32) - 127.5) / 128.0
        x = np.transpose(x, (2,0,1))[None, ...]  # 1x3x112x112
        return x

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        if self.use_insightface:
            # crop by detection for robustness; if no face, do center-crop resize
            dets = self.app.get(bgr)
            if len(dets) > 0 and hasattr(dets[0], "normed_embedding"):
                # pick largest face
                det = max(dets, key=lambda d: (d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1]))
                emb = det.normed_embedding.astype(np.float32)
                return emb
            else:
                # fallback: raw resize to backbone input then get embedding from app.model (if exposed)
                x = self._preproc_arc(bgr, self.size)
                # This path is rare; keeping compatibility:
                from insightface.model_zoo import get_model
                # default ArcFace r100
                m = get_model('arcface_r100_v1')
                m.prepare(ctx_id=0 if cv2.ocl.haveOpenCL() else -1)
                emb = m.get_feat(x[0].transpose(1,2,0)[:,:,::-1])  # expects BGR->???; adjust if needed
                return emb.astype(np.float32)
        else:
            x = self._preproc_arc(bgr, self.size)
            emb = self.ort_sess.run(None, {self.input_name: x})[0]
            emb = emb.squeeze().astype(np.float32)
            # normalize to unit
            n = np.linalg.norm(emb) + 1e-8
            return emb / n

# ---------- Squeezers ----------
def bit_depth_squeeze(bgr: np.ndarray, bits: int) -> np.ndarray:
    if bits >= 8: 
        return bgr.copy()
    shift = 8 - bits
    out = (bgr >> shift) << shift
    return out

def median_squeeze(bgr: np.ndarray, k: int) -> np.ndarray:
    k = max(3, k | 1)
    return cv2.medianBlur(bgr, k)

def bilateral_squeeze(bgr: np.ndarray, d=5, sigma_color=50, sigma_space=7) -> np.ndarray:
    return cv2.bilateralFilter(bgr, d, sigma_color, sigma_space)

# Optional: Non-Local Means (slower, stronger denoise)
def nlm_squeeze(bgr: np.ndarray, h=10, tmpl=7, search=21) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(bgr, None, h, h, tmpl, search)

# ---------- Cosine divergence ----------
def cos_div(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    a /= (np.linalg.norm(a) + 1e-8)
    b /= (np.linalg.norm(b) + 1e-8)
    cos = float(np.dot(a, b))
    return max(0.0, 1.0 - cos)

# ---------- Main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img", required=True, help="input image path")
    #p.add_argument("--tau", type=float, default=0.08, help="adv threshold on max cosine divergence")
    p.add_argument("--tau", type=float, default=0.16476, help="adv threshold on max cosine divergence (tuned)")
    p.add_argument("--save-report", default=None, help="path to save txt report (optional)")
    p.add_argument("--denoise-out", default=None, help="if set, saves best cleaned image here when flagged OR always if --force-save")
    p.add_argument("--force-save", action="store_true", help="always save best squeezer result, not only when flagged")
    p.add_argument("--use-nlm", action="store_true", help="include NLM squeezer (slower, stronger)")
    p.add_argument("--arc-model", default=os.environ.get("ARC_MODEL", ""), help="optional .onnx model path")
    args = p.parse_args()

    img_path = Path(args.img)
    assert img_path.exists(), f"not found: {img_path}"

    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    assert bgr is not None, f"cannot read: {img_path}"

    # embedder
    onnx_path = args.arc_model if args.arc_model else None
    embedder = ArcEmbedder(onnx_path=onnx_path)

    # original embedding
    emb0 = embedder(bgr)

    # build squeezers
    squeezers = []
    # bit-depths
    for bits in (5, 4):
        squeezers.append(("bit", {"bits": bits}))
    # median
    for k in (3, 5):
        squeezers.append(("median", {"k": k}))
    # bilateral
    squeezers.append(("bilateral", {"d": 5, "sigma_color": 40, "sigma_space": 7}))
    # nlm (optional)
    if args.use_nlm:
        squeezers.append(("nlm", {"h": 8, "tmpl": 7, "search": 21}))

    # evaluate
    records = []
    best = (None, -1e9, None)  # (name, divergence, img)
    for name, kw in squeezers:
        if name == "bit":
            img = bit_depth_squeeze(bgr, **kw)
        elif name == "median":
            img = median_squeeze(bgr, **kw)
        elif name == "bilateral":
            img = bilateral_squeeze(bgr, **kw)
        elif name == "nlm":
            img = nlm_squeeze(bgr, **kw)
        else:
            continue

        emb = embedder(img)
        d = cos_div(emb0, emb)
        records.append((name, kw, d))
        if d > best[1]:
            best = (f"{name}:{kw}", d, img)

    max_div = best[1]
    flagged = max_div >= args.tau

    # report
    lines = []
    lines.append(f"image: {img_path.name}")
    lines.append(f"tau: {args.tau:.4f}")
    lines.append("divergences:")
    for name, kw, d in records:
        lines.append(f"  - {name} {kw} : {d:.5f}")
    lines.append(f"max_div: {max_div:.5f}")
    lines.append(f"flagged_adv: {int(flagged)}")

    report = "\n".join(lines)
    print(report)

    if args.save_report:
        Path(args.save_report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_report, "w", encoding="utf-8") as f:
            f.write(report + "\n")

    # save cleaned if requested
    if args.denoise_out and (flagged or args.force_save):
        Path(args.denoise_out).parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(args.denoise_out), best[2])
        if not ok:
            print(f"[warn] failed to save denoised to {args.denoise_out}")

if __name__ == "__main__":
    main()
