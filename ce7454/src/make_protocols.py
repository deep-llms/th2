#!/usr/bin/env python3
"""Build post-processed copies of the held-out val split and of dev for robustness
evaluation, using only published evaluation protocols (fixed seeds, deterministic):

  jpeg95/75/50  : JPEG re-compression (PIL) at fixed quality (Wang et al. 2020 grid)
  blur1/blur2   : Gaussian blur, sigma 1 / 2 (Wang et al. 2020 grid)
  down075/down05: bicubic downscale x0.75 / x0.5 (B-Free, SPAI, Rajan et al. grids)
  vipcup        : random crop of side U[5/8, 1] x size at a random position -> resize to
                  200x200 (bicubic) -> JPEG quality U{65..100} (IEEE VIP Cup 2022 protocol
                  as described in Cozzolino et al. CVPRW 2024 Sec. 3 / Corvi et al. 2023)
  social        : downscale x U[0.7, 1] (bicubic) -> JPEG quality U{70..100}
                  (B-Free CVPR 2025 social-network simulation)

Output: cache/proto_<split>_<name>.pkl = list of uint8 HxWx3 arrays (val order follows
stratified_split(train labels, 0.1, 42)).
"""
import io
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import load_meta, stratified_split  # noqa: E402

CACHE = "/disk/thuat/CE7454/cache"
PROTOCOLS = ["jpeg95", "jpeg75", "jpeg50", "blur1", "blur2", "down075", "down05", "vipcup", "social"]


def jpeg(a, q):
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "JPEG", quality=int(q))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def resize(a, w, h):
    return np.asarray(Image.fromarray(a).resize((int(w), int(h)), Image.BICUBIC))


def blur(a, s):
    out = np.array(a)
    for c in range(3):
        gaussian_filter(a[:, :, c], output=out[:, :, c], sigma=s)
    return out


def apply(name, a, rng):
    h, w = a.shape[:2]
    if name.startswith("jpeg"):
        return jpeg(a, int(name[4:]))
    if name.startswith("blur"):
        return blur(a, float(name[4:]))
    if name == "down075":
        return resize(a, w * 0.75, h * 0.75)
    if name == "down05":
        return resize(a, w * 0.5, h * 0.5)
    if name == "vipcup":
        side = int(round(rng.uniform(5 / 8, 1.0) * min(h, w)))
        y, x = rng.randint(0, h - side + 1), rng.randint(0, w - side + 1)
        return jpeg(resize(np.ascontiguousarray(a[y:y + side, x:x + side]), 200, 200), rng.randint(65, 101))
    if name == "social":
        s = rng.uniform(0.7, 1.0)
        return jpeg(resize(a, round(w * s), round(h * s)), rng.randint(70, 101))
    raise ValueError(name)


def job(args):
    split, name, idx, seed = args
    arr = np.load(os.path.join(CACHE, f"{split}_images.npy"), mmap_mode="r")
    rng = np.random.RandomState(seed)
    return [apply(name, np.array(arr[i]), rng) for i in idx]


def main():
    labels = np.asarray(load_meta(CACHE, "train")["labels"])
    _, va = stratified_split(labels, 0.1, 42)
    sets = {"val": ("train", va), "dev": ("dev", np.arange(len(load_meta(CACHE, "dev")["filenames"])))}
    for pi, name in enumerate(PROTOCOLS):
        for si, (tag, (split, idx)) in enumerate(sets.items()):
            out = os.path.join(CACHE, f"proto_{tag}_{name}.pkl")
            if os.path.exists(out):
                continue
            chunks = np.array_split(idx, 8)
            seeds = [1000 * (pi + 1) + 100 * si + k for k in range(len(chunks))]
            with Pool(4) as pool:
                parts = pool.map(job, [(split, name, c, s) for c, s in zip(chunks, seeds)])
            imgs = [im for p in parts for im in p]
            pickle.dump(imgs, open(out, "wb"), protocol=4)
            print(f"{tag} {name}: {len(imgs)} images, e.g. {imgs[0].shape}", flush=True)


if __name__ == "__main__":
    main()
