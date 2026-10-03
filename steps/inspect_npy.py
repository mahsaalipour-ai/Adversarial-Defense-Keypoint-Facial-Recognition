# inspect_npy.py
import numpy as np
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_npy.py <path/to/file.npy>")
        return

    p = Path(sys.argv[1]).expanduser().resolve()
    if not p.exists():
        print("File not found:", p)
        return

    a = np.load(p, allow_pickle=False)
    print("file:", p)
    print("shape:", a.shape, "dtype:", a.dtype)
    if a.size:
        try:
            print("min:", a.min(), "max:", a.max())
        except Exception:
            pass
    # نشان دادن 20 ردیف اول (یا کمتر)
    if a.ndim == 1:
        print("first values:", a[:20])
    elif a.ndim == 2:
        print("first rows:\n", a[:20])
    else:
        # reshape for preview if higher-dim
        flat = a.reshape(-1, a.shape[-1]) if a.ndim > 1 else a.flatten()
        print("preview (first 20 rows):\n", flat[:20])

if __name__ == "__main__":
    main()


#for run
#python inspect_npy.py "D:\PayanNameh\paper-structure\deep-keypoints-attack\data\step2\000001_dst_pts_step2.npy"
