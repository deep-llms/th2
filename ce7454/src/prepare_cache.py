#!/usr/bin/env python3
"""Decode all JPEGs once into uint8 memmap arrays for fast training.

Creates in --cache-dir:
  train_images.npy  (N,512,512,3) uint8   + train_meta.json (filenames, labels)
  dev_images.npy    (N,512,512,3) uint8   + dev_meta.json   (filenames)
  test_images.npy   (N,512,512,3) uint8   + test_meta.json  (filenames)
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image

RES = 512


def decode_one(path):
    im = Image.open(path).convert("RGB")
    a = np.asarray(im, dtype=np.uint8)
    assert a.shape == (RES, RES, 3), f"{path}: {a.shape}"
    return a


def build_split(name, root, filenames, cache_dir, workers):
    out_path = os.path.join(cache_dir, f"{name}_images.npy")
    arr = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.uint8, shape=(len(filenames), RES, RES, 3)
    )
    paths = [os.path.join(root, f) for f in filenames]
    with Pool(workers) as pool:
        for i, img in enumerate(pool.imap(decode_one, paths, chunksize=64)):
            arr[i] = img
            if (i + 1) % 2000 == 0:
                print(f"  {name}: {i + 1}/{len(filenames)}")
    arr.flush()
    print(f"  {name}: wrote {out_path} shape={arr.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/disk/thuat/CE7454/resources")
    ap.add_argument("--cache-dir", default="/disk/thuat/CE7454/cache")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    train_root = os.path.join(args.data_root, "NTU_CE7454_2026_dev_phase/dev_phase/train")
    dev_root = os.path.join(args.data_root, "NTU_CE7454_2026_dev_phase/dev_phase/dev")
    test_root = os.path.join(args.data_root, "NTU_CE7454_2026_Aug_test/test")

    # Train: sort by index for a stable, reproducible order.
    train_files = sorted(os.listdir(train_root), key=lambda f: int(f.split("_")[0]))
    labels = [int(f[:-4].split("_")[1]) for f in train_files]
    with open(os.path.join(args.cache_dir, "train_meta.json"), "w") as fp:
        json.dump({"filenames": train_files, "labels": labels}, fp)
    build_split("train", train_root, train_files, args.cache_dir, args.workers)

    for name, root in [("dev", dev_root), ("test", test_root)]:
        files = sorted(os.listdir(root), key=lambda f: int(f.split(".")[0]))
        with open(os.path.join(args.cache_dir, f"{name}_meta.json"), "w") as fp:
            json.dump({"filenames": files}, fp)
        build_split(name, root, files, args.cache_dir, args.workers)

    print("done")


if __name__ == "__main__":
    main()
