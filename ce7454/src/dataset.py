"""Datasets over the uint8 memmap caches produced by prepare_cache.py.

Train-time pipeline (all options off by default except crop + geometric flips):
  1. optional paper-specific processing augmentation on the full image, in the
     order used by the original code:
       rrc_p    : RandomResizedCrop (Corvi et al. ICASSP 2023 / GRIP code: scale
                  (0.08, 1), ratio (3/4, 4/3), bilinear, output rrc_size)
       grip     : Gaussian blur (scipy, per channel, sigma ~ U[0, 3]) then JPEG
                  (quality ~ U{30..100}, cv2 or PIL), each with its probability
                  (Wang et al. CVPR 2020 / Gragnaniello et al. ICME 2021)
       frank_p  : Frank et al. ICML 2020 perturbations: one of {noise, blur, jpeg,
                  crop+bicubic-rescale}, applied with probability frank_p
       he_scale : He et al. 2016 scale augmentation (shorter side ~ U[256, 480])
  2. geometric: "dihedral" (h/v flips + rot90), "rot90_hflip" (GRIP: rot90 then
     hflip), "hflip", or "none"
  3. random crop of `crop` pixels (crop >= image size: no crop)
  4. optional He et al. 2016 / AlexNet PCA colour lighting (pca_std)
For speed, when a small crop is taken and no resize happens (RandomResizedCrop is drawn
first and only then the full image is used), the processing is
applied to a random region of crop+32 px instead of the whole image; this is
statistically equivalent (random crop position, random JPEG-grid phase, blur
border margin > 3 sigma).
Eval: full 512x512 image, no augmentation.
"""
import io
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

RES = 512

# ImageNet PCA lighting (AlexNet), as in fb.resnet.torch used by He et al. 2016.
_PCA_EIGVAL = np.array([0.2175, 0.0188, 0.0045], np.float32)
_PCA_EIGVEC = np.array([[-0.5675, 0.7192, 0.4009],
                        [-0.5808, -0.0045, -0.8140],
                        [-0.5836, -0.6948, 0.4203]], np.float32)


def load_meta(cache_dir, split):
    with open(os.path.join(cache_dir, f"{split}_meta.json")) as fp:
        return json.load(fp)


def stratified_split(labels, val_frac=0.1, seed=42):
    """Return (train_idx, val_idx), stratified by label."""
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_frac))
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def jpeg_cv2(img, q):
    ok, enc = cv2.imencode(".jpg", np.ascontiguousarray(img[:, :, ::-1]), [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
    return cv2.imdecode(enc, 1)[:, :, ::-1]


def jpeg_pil(img, q):
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img)).save(buf, "JPEG", quality=int(q))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def random_resized_crop(img, size, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3), interp=Image.BILINEAR):
    """torchvision.transforms.RandomResizedCrop.get_params + bilinear resize."""
    h, w = img.shape[:2]
    area = h * w
    for _ in range(10):
        target = area * np.random.uniform(*scale)
        ar = np.exp(np.random.uniform(np.log(ratio[0]), np.log(ratio[1])))
        cw = int(round(np.sqrt(target * ar)))
        ch = int(round(np.sqrt(target / ar)))
        if 0 < cw <= w and 0 < ch <= h:
            y = np.random.randint(0, h - ch + 1)
            x = np.random.randint(0, w - cw + 1)
            break
    else:  # fallback: central crop
        ch, cw, y, x = h, w, 0, 0
    pil = Image.fromarray(np.ascontiguousarray(img[y:y + ch, x:x + cw]))
    return np.asarray(pil.resize((size, size), interp))


def frank_perturb(img):
    """Frank et al. 2020 (create_perturbed_imagedata.py, 'combined' mode): one of four
    perturbations. JPEG uses the paper's quality U[10, 75) (the released code passes the
    quality as a flag id, so it effectively encodes at the cv2 default)."""
    kind = np.random.randint(4)
    if kind == 0:  # noise, std ~ U[5, 20]
        std = np.random.uniform(5.0, 20.0)
        return np.clip(img.astype(np.float64) + std * np.random.randn(*img.shape), 0, 255).astype(np.uint8)
    if kind == 1:  # blur, kernel in {3,5,7,9}, sigmaX = cv2.BORDER_DEFAULT (= 4) as in the code
        k = int(np.random.choice([3, 5, 7, 9]))
        return cv2.GaussianBlur(np.ascontiguousarray(img), (k, k), sigmaX=cv2.BORDER_DEFAULT)
    if kind == 2:
        return jpeg_cv2(img, np.random.randint(10, 75))
    pct = np.random.uniform(0.05, 0.2)  # crop 5-20%, bicubic back to the original size
    h, w = img.shape[:2]
    yc, xc = int(h * pct * 0.5), int(w * pct * 0.5)
    return cv2.resize(np.ascontiguousarray(img[yc:h - yc, xc:w - xc]), (w, h), interpolation=cv2.INTER_CUBIC)


CF_INTERPS = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS, Image.HAMMING, Image.BOX]


def social_cf(img, cf_interp=False):
    """Community Forensics (Park & Owens, CVPR 2025) processing augmentation, processing ops only:
    resize by U[225,275]/256 (bilinear or bicubic), RandomResizedCrop scale (0.9,1) back to the same
    size, JPEG quality U{75..100} (PIL or cv2); each op applied n ~ U{0,1,2} times, random order."""
    ops = [k for k in ("resize", "rrc", "jpeg") for _ in range(np.random.randint(3))]
    np.random.shuffle(ops)
    for op in ops:
        h, w = img.shape[:2]
        if op == "resize":
            f = np.random.randint(225, 276) / 256.0
            interp = Image.BILINEAR if np.random.rand() < 0.5 else Image.BICUBIC
            img = np.asarray(Image.fromarray(np.ascontiguousarray(img)).resize(
                (max(1, int(round(w * f))), max(1, int(round(h * f)))), interp))
        elif op == "rrc":
            # cf_interp: interpolation drawn per call from the 6 used in the official CF code
            img = random_resized_crop(img, min(h, w), scale=(0.9, 1.0),
                                      interp=CF_INTERPS[np.random.randint(6)] if cf_interp else Image.BILINEAR)
        else:
            q = np.random.randint(75, 101)
            img = (jpeg_cv2 if np.random.rand() < 0.5 else jpeg_pil)(img, q)
    return img


BFREE_INTERPS = [Image.BILINEAR, Image.BICUBIC, Image.LANCZOS, Image.BOX]


def social_bfree(img):
    """B-Free (Guillaro et al., CVPR 2025, Supp. F) social-network simulation: downscale by a factor
    U[0.7, 1] (interpolation drawn from bilinear/bicubic/Lanczos/box) then JPEG quality U{70..100}
    (PIL or cv2, 4:2:0). Resampling by a non-integer factor also re-grids the image, so the result
    is close to a once-compressed image of a slightly smaller original."""
    h, w = img.shape[:2]
    f = np.random.uniform(0.7, 1.0)
    img = np.asarray(Image.fromarray(np.ascontiguousarray(img)).resize(
        (max(1, int(round(w * f))), max(1, int(round(h * f)))), BFREE_INTERPS[np.random.randint(4)]))
    q = np.random.randint(70, 101)
    return (jpeg_cv2 if np.random.rand() < 0.5 else jpeg_pil)(img, q)


def canonicalize(a, mode):
    """Fixed, deterministic input canonicalisation applied identically to every image (training and inference):
    'jpegQ' = re-encode with PIL JPEG quality Q and 4:2:0 chroma (JPEG input-transformation, Guo et al. ICLR 2018 /
    Das et al. 2017); 'luma' = BT.601 luminance replicated to 3 channels; 'lumajpegQ' = luminance then JPEG Q."""
    if mode == "none":
        return a
    if mode.startswith("luma"):
        yv = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).round().clip(0, 255).astype(np.uint8)
        a = np.repeat(yv[..., None], 3, axis=2)
        mode = mode[4:] or "none"
        if mode == "none":
            return a
    if mode.startswith("jpeg"):
        b = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(a)).save(b, "JPEG", quality=int(mode[4:]), subsampling=2)
        b.seek(0)
        return np.asarray(Image.open(b).convert("RGB"))
    raise ValueError(mode)


def newbatch_match(a):
    """Train-side appearance matching to the new image batch (dev/test share it; measured label-free): every class there is
    sharper (x1.24-1.35 luma Laplacian), noisier (x1.4-1.5 MAD) and less saturated. Applied to the whole training image
    before any view is formed: unsharp mask (radius 0.8, amount U[30,70]%), saturation x U[0.8,0.9], Gaussian noise
    sigma U[0,2.5] grey levels. Inference is unchanged."""
    im = Image.fromarray(np.ascontiguousarray(a)).filter(ImageFilter.UnsharpMask(radius=0.8, percent=int(np.random.uniform(30, 70)), threshold=0))
    x = np.asarray(im, np.float32)
    g = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])[..., None]
    x = g + np.random.uniform(0.8, 0.9) * (x - g)
    x = x + np.random.randn(*x.shape).astype(np.float32) * np.random.uniform(0, 2.5)
    return np.clip(x, 0, 255).round().astype(np.uint8)


def newbatch_match_luma(a):
    """Variant of newbatch_match after the research agent's critique: the residual noise is added to LUMINANCE only
    (identical in R, G, B), so no chroma high-frequency energy is added (the test images have weak chroma HF). Unsharp
    mask radius 0.8 amount U[30,70]%, saturation x U[0.8,0.9], luma noise sigma U[0,2.5]."""
    im = Image.fromarray(np.ascontiguousarray(a)).filter(ImageFilter.UnsharpMask(radius=0.8, percent=int(np.random.uniform(30, 70)), threshold=0))
    x = np.asarray(im, np.float32)
    g = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])[..., None]
    x = g + np.random.uniform(0.8, 0.9) * (x - g)
    x = x + np.random.randn(x.shape[0], x.shape[1], 1).astype(np.float32) * np.random.uniform(0, 2.5)
    return np.clip(x, 0, 255).round().astype(np.uint8)


def rechain(a):
    """Minimal re-encoding augmentation on the full image: a near-identity crop (496-512 px) resized back to 512 with a
    random kernel, then PIL JPEG at quality U{85..95} with 4:2:0 chroma (the encoding seen in the test headers)."""
    n = a.shape[0]
    s = np.random.randint(n - 16, n + 1)
    y, x = np.random.randint(0, n - s + 1, 2)
    im = Image.fromarray(np.ascontiguousarray(a[y:y + s, x:x + s]))
    if s != n:
        im = im.resize((n, n), [Image.BILINEAR, Image.BICUBIC, Image.LANCZOS][np.random.randint(3)])
    b = io.BytesIO()
    im.save(b, "JPEG", quality=int(np.random.randint(85, 96)), subsampling=2)
    b.seek(0)
    return np.asarray(Image.open(b).convert("RGB"))


class CachedImageDataset(Dataset):
    """Serves uint8 CHW tensors; normalization happens on GPU in the train loop."""

    def __init__(self, cache_dir, split, indices=None, labels=None, train=False, crop=256,
                 jpeg_p=0.0, jpeg_qmin=60, jpeg_qmax=100, blur_p=0.0, blur_smax=1.0,
                 gray_p=0.0, jitter_p=0.0, jitter_s=0.1, flips="dihedral",
                 grip_blur_p=0.0, grip_jpeg_p=0.0, rrc_p=0.0, rrc_size=256, frank_p=0.0,
                 he_scale=False, pca_std=0.0, grip_blur_smax=3.0, grip_jpeg_qmin=30,
                 grip_jpeg_qmax=100, rrc_scale_min=0.08, social_p=0.0, views=1, social_cf_interp=False, social_region=False,
                 social_mode="cf", sharpen_p=0.0, sharpen_mode="composite",
                 soft_p=0.0, soft_smax=1.0, canon="none", noise_p=0.0, noise_smax=4.0, desat_p=0.0, desat_min=0.6, rechain_p=0.0, match_p=0.0, match_mode="rgb", shuffle_p=0.0, shuffle_tile=32, flat_p=0.0, flat_k=8, hfgain_p=0.0, hfgain_max=1.6, crop_upscale=False, chroma420_p=0.0):
        self.match_p, self.match_mode = match_p, match_mode
        self.shuffle_p, self.shuffle_tile = shuffle_p, shuffle_tile
        self.canon = canon
        self.rechain_p = rechain_p
        self.desat_p, self.desat_min = desat_p, desat_min
        self.noise_p, self.noise_smax = noise_p, noise_smax
        self.path = os.path.join(cache_dir, f"{split}_images.npy")
        self.arr = None  # opened lazily per worker (mmap is not fork-safe to share)
        meta = load_meta(cache_dir, split)
        n = len(meta["filenames"])
        self.indices = np.arange(n) if indices is None else np.asarray(indices)
        self.labels = None if labels is None else np.asarray(labels)
        self.train = train
        self.crop = crop
        self.flat_p, self.flat_k = flat_p, flat_k
        self.hfgain_p, self.hfgain_max = hfgain_p, hfgain_max
        self.crop_upscale = crop_upscale
        self.chroma420_p = chroma420_p
        assert flips in ("dihedral", "hflip", "rot90_hflip", "none")
        self.flips = flips
        # Legacy robustness options (earlier experiments; applied to the crop).
        self.jpeg_p, self.jpeg_qmin, self.jpeg_qmax = jpeg_p, jpeg_qmin, jpeg_qmax
        self.blur_p, self.blur_smax = blur_p, blur_smax
        self.gray_p, self.jitter_p, self.jitter_s = gray_p, jitter_p, jitter_s
        # Paper-recipe processing options (see module docstring).
        self.grip_blur_p, self.grip_jpeg_p = grip_blur_p, grip_jpeg_p
        self.rrc_p, self.rrc_size = rrc_p, rrc_size
        self.grip_blur_smax = grip_blur_smax
        self.grip_jpeg_qmin, self.grip_jpeg_qmax = grip_jpeg_qmin, grip_jpeg_qmax
        self.rrc_scale_min = rrc_scale_min
        self.social_p = social_p
        self.views = views
        self.social_cf_interp = social_cf_interp
        self.social_region = social_region
        self.social_mode = social_mode
        self.sharpen_p, self.sharpen_mode = sharpen_p, sharpen_mode
        self.soft_p, self.soft_smax = soft_p, soft_smax
        self.frank_p = frank_p
        self.he_scale = he_scale
        self.pca_std = pca_std

    def __len__(self):
        return len(self.indices)

    def _get_arr(self):
        if self.arr is None:
            self.arr = np.load(self.path, mmap_mode="r")
        return self.arr

    def _process(self, img, do_rrc=False, do_social=False):
        """Paper processing augmentation on the (region of the) image, original order.
        `do_rrc` is drawn by the caller (probability rrc_p) so that the full image is only
        loaded when it is actually resized."""
        # Sharpness jitter (before encoding): sharpen with prob sharpen_p, otherwise mild Gaussian blur
        # sigma U[0, soft_smax] with prob soft_p (PatchCraft sigma 0-1; SPAI <= 0.5). With soft_p = 0 this
        # is exactly the earlier sharpen-only behaviour (same random draws).
        u = np.random.rand() if (self.sharpen_p > 0 or self.soft_p > 0) else 1.0
        if self.soft_p > 0 and self.sharpen_p <= u < self.sharpen_p + self.soft_p:
            img = np.array(img)
            sig = np.random.uniform(0.0, self.soft_smax)
            for ch in range(3):
                gaussian_filter(img[:, :, ch], output=img[:, :, ch], sigma=sig)
        if self.sharpen_p > 0 and u < self.sharpen_p:
            # Sharpening before (re-)encoding, as in camera / platform pipelines.
            pil = Image.fromarray(np.ascontiguousarray(img))
            if self.sharpen_mode == "composite":   # Composite DA 2025 / torchvision RandomAdjustSharpness(2)
                img = np.asarray(ImageEnhance.Sharpness(pil).enhance(2.0))
            else:                                  # random unsharp mask (camera-style)
                img = np.asarray(pil.filter(ImageFilter.UnsharpMask(radius=np.random.uniform(0.5, 1.5),
                                                                  percent=int(np.random.uniform(20, 150)),
                                                                  threshold=0)))
        if self.desat_p > 0 and np.random.rand() < self.desat_p:
            # chroma reduction for every class: blend towards BT.601 luma with factor s ~ U[desat_min, 1]
            a = np.asarray(img, np.float32)
            g = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])[..., None]
            sf = np.random.uniform(self.desat_min, 1.0)
            img = np.clip(g + sf * (a - g), 0, 255).round().astype(np.uint8)
        if self.hfgain_p > 0 and np.random.rand() < self.hfgain_p:
            # high-frequency gain jitter (our diagnosis: the model reads the global HF energy level as real evidence):
            # scale the detail band (image - Gaussian(sigma 1) low-pass) by g ~ log-uniform[1/hfgain_max, hfgain_max],
            # for every class alike; keeps the residual's structure (signs, correlations), changes only its gain
            a = np.asarray(img, np.float32)
            lp = np.stack([gaussian_filter(a[..., ch], sigma=1.0) for ch in range(3)], -1)
            gg = float(np.exp(np.random.uniform(-np.log(self.hfgain_max), np.log(self.hfgain_max))))
            img = np.clip(lp + gg * (a - lp), 0, 255).round().astype(np.uint8)
        if self.noise_p > 0 and np.random.rand() < self.noise_p:
            # additive Gaussian noise, sigma ~ U[0, noise_smax] grey levels, before (re-)encoding, for every class alike:
            # breaks the noise-level cue measured in the TRAINING data (real images are the noisiest class)
            sig = np.random.uniform(0.0, self.noise_smax)
            img = np.clip(np.asarray(img, np.float32) + np.random.randn(*np.shape(img)).astype(np.float32) * sig, 0, 255).round().astype(np.uint8)
        if do_social:
            img = social_bfree(img) if self.social_mode == "bfree" else social_cf(img, self.social_cf_interp)
        if self.he_scale:
            s = np.random.randint(256, 481)
            img = np.asarray(Image.fromarray(np.ascontiguousarray(img)).resize((s, s), Image.BILINEAR))
        if do_rrc:
            img = random_resized_crop(img, self.rrc_size, scale=(self.rrc_scale_min, 1.0))
        if self.grip_blur_p > 0 and np.random.rand() < self.grip_blur_p:
            img = np.array(img)
            sig = np.random.uniform(0.0, self.grip_blur_smax)
            for c in range(3):
                gaussian_filter(img[:, :, c], output=img[:, :, c], sigma=sig)
        if self.grip_jpeg_p > 0 and np.random.rand() < self.grip_jpeg_p:
            q = np.random.randint(self.grip_jpeg_qmin, self.grip_jpeg_qmax + 1)
            img = (jpeg_cv2 if np.random.rand() < 0.5 else jpeg_pil)(img, q)
        if self.frank_p > 0 and np.random.rand() < self.frank_p:
            img = frank_perturb(img)
        return img

    def __getitem__(self, i):
        idx = self.indices[i]
        img = self._get_arr()[idx]  # (512,512,3) uint8 mmap view
        if self.canon != "none":
            img = canonicalize(np.array(img), self.canon)   # fixed input canonicalisation (train and eval alike)
        if self.train and self.match_p > 0 and np.random.rand() < self.match_p:
            img = (newbatch_match_luma if self.match_mode == "luma" else newbatch_match)(np.array(img))   # training only
        if self.train and self.rechain_p > 0 and np.random.rand() < self.rechain_p:
            img = rechain(np.array(img))                    # minimal re-encoding augmentation (training only)
        if self.train and self.chroma420_p > 0 and np.random.rand() < self.chroma420_p:
            # chroma-subsampling augmentation (training only, all classes): JPEG quality 100 with 4:2:0, i.e. the colour
            # channels at half resolution with negligible luma quantisation
            b = io.BytesIO(); Image.fromarray(np.ascontiguousarray(np.array(img))).save(b, "JPEG", quality=100, subsampling=2)
            b.seek(0); img = np.asarray(Image.open(b).convert("RGB"))
        if not self.train:
            t = torch.from_numpy(np.ascontiguousarray(np.array(img).transpose(2, 0, 1)))
        elif self.views == 1:
            t = self._train_view(img, True)
            if self.shuffle_p > 0 and np.random.rand() < self.shuffle_p:
                t = self._patch_shuffle(t)
        else:
            # Consistency training (AugMix-style): view 0 has only the geometric augmentation,
            # views 1.. additionally get the processing augmentation; each view has its own crop.
            vs = [self._train_view(img, v > 0) for v in range(self.views)]
            if self.shuffle_p > 0:
                vs = [self._patch_shuffle(v_) if np.random.rand() < self.shuffle_p else v_ for v_ in vs]
            t = torch.stack(vs)
        if self.labels is not None:
            return t, int(self.labels[i])
        return t, idx

    def _patch_shuffle(self, t):
        """Patch-shuffle (Zheng et al., NeurIPS 2024): permute the non-overlapping tiles of a (C,H,W) view."""
        k = self.shuffle_tile
        C, H, W = t.shape
        gh, gw = H // k, W // k
        tiles = t[:, :gh * k, :gw * k].reshape(C, gh, k, gw, k).permute(1, 3, 0, 2, 4).reshape(gh * gw, C, k, k)
        tiles = tiles[torch.from_numpy(np.random.permutation(gh * gw))]
        out = t.clone()
        out[:, :gh * k, :gw * k] = tiles.reshape(gh, gw, C, k, k).permute(2, 0, 3, 1, 4).reshape(C, gh * k, gw * k)
        return out

    def _train_view(self, img, with_processing):
        if True:
            c = self.crop
            region_crop = None
            processing = with_processing and (self.grip_blur_p or self.grip_jpeg_p or self.rrc_p
                                              or self.frank_p or self.he_scale or self.social_p
                                              or self.sharpen_p or self.soft_p or self.noise_p or self.desat_p or self.hfgain_p)
            if processing:
                do_rrc = self.rrc_p > 0 and np.random.rand() < self.rrc_p
                # drawn only when enabled, so earlier configurations keep identical random streams
                do_social = self.social_p > 0 and np.random.rand() < self.social_p
                resizes = do_rrc or do_social or self.he_scale or self.frank_p > 0
                # social_region: the CF chain shrinks at most to 0.879^2 (resize) x 0.949^2 (RRC) of the
                # input side, so a random region of 1.45*crop (+16) still contains a full crop after it;
                # statistically equivalent to processing the whole image, ~2-4x less CPU.
                r_social = int(np.ceil(1.45 * c)) + 16
                region_crop = None
                if do_social and self.social_region and not (do_rrc or self.he_scale or self.frank_p > 0) \
                        and r_social < RES:
                    # Draw the final crop position uniformly over the whole image first (as when the
                    # full image is processed), then process a region around it and cut the crop at
                    # the corresponding place; a uniformly placed region + a uniform crop inside it
                    # would over-sample the image centre.
                    py, px = np.random.randint(0, RES - c + 1, 2)
                    y = int(np.clip(py - (r_social - c) // 2, 0, RES - r_social))
                    x = int(np.clip(px - (r_social - c) // 2, 0, RES - r_social))
                    region_crop = (py - y, px - x, r_social)
                    img = np.array(img[y:y + r_social, x:x + r_social])
                elif not resizes and c + 32 < RES:  # process only a random region (see docstring)
                    r = c + 32
                    y, x = np.random.randint(0, RES - r + 1, 2)
                    img = np.array(img[y:y + r, x:x + r])
                else:
                    img = np.array(img)
                img = self._process(img, do_rrc, do_social)
                if region_crop is not None:
                    oy, ox, r0 = region_crop
                    h, w = img.shape[:2]
                    cy = int(np.clip(round(oy * h / r0), 0, h - c))
                    cx = int(np.clip(round(ox * w / r0), 0, w - c))
                    img = img[cy:cy + c, cx:cx + c]  # geometric ops below commute with the crop
            # geometric augmentation
            if self.flips == "rot90_hflip":
                k = np.random.randint(4)
                if k:
                    img = np.rot90(img, k)
                if np.random.rand() < 0.5:
                    img = img[:, ::-1]
            elif self.flips in ("dihedral", "hflip"):
                if np.random.rand() < 0.5:
                    img = img[:, ::-1]
                if self.flips == "dihedral":
                    if np.random.rand() < 0.5:
                        img = img[::-1, :]
                    k = np.random.randint(4)
                    if k:
                        img = np.rot90(img, k)
            # random crop (no crop if the image is not larger than `crop`)
            h, w = img.shape[:2]
            if self.crop_upscale and (h < c or w < c):
                # large crops (> 395 px): a processed view shrunk below the crop by the CF chain is resized back up
                # (bilinear, factor <= 1.14) so every view has the crop size; off by default
                img = np.asarray(Image.fromarray(np.ascontiguousarray(img)).resize((max(w, c), max(h, c)), Image.BILINEAR))
                h, w = img.shape[:2]
            if c < h or c < w:
                if self.flat_p > 0 and np.random.rand() < self.flat_p:
                    # flat-region crop sampling (our diagnosis: real->fake errors are images dominated by flat
                    # regions): among flat_k uniformly drawn positions keep the one with the lowest mean luma
                    # gradient magnitude (integral image), so the crop's evidence must come from flat content.
                    a = np.asarray(img, np.float32)
                    g = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
                    gm = np.zeros((h + 1, w + 1), np.float64)
                    gm[1:, 1:] = (np.abs(np.diff(g, axis=0, append=g[-1:])) + np.abs(np.diff(g, axis=1, append=g[:, -1:]))).cumsum(0).cumsum(1)
                    ys = np.random.randint(0, h - c + 1, self.flat_k)
                    xs = np.random.randint(0, w - c + 1, self.flat_k)
                    sc = gm[ys + c, xs + c] - gm[ys, xs + c] - gm[ys + c, xs] + gm[ys, xs]
                    j = int(np.argmin(sc))
                    y, x = int(ys[j]), int(xs[j])
                else:
                    y = np.random.randint(0, h - c + 1)
                    x = np.random.randint(0, w - c + 1)
                img = img[y:y + c, x:x + c]
            img = np.array(img)  # copy out of the mmap / make contiguous
            if self.pca_std > 0:
                alpha = np.random.normal(0, self.pca_std, 3).astype(np.float32)
                rgb = _PCA_EIGVEC @ (alpha * _PCA_EIGVAL) * 255.0
                img = np.clip(img.astype(np.float32) + rgb, 0, 255).astype(np.uint8)
            if self.gray_p > 0 and np.random.rand() < self.gray_p:
                g = (img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32))
                img = np.repeat(np.clip(g, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
            if self.jitter_p > 0 and np.random.rand() < self.jitter_p:
                pil = Image.fromarray(np.ascontiguousarray(img))
                s = self.jitter_s
                for enh in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
                    pil = enh(pil).enhance(1.0 + np.random.uniform(-s, s))
                img = np.asarray(pil)
            if self.blur_p > 0 and np.random.rand() < self.blur_p:
                sigma = np.random.uniform(0.1, self.blur_smax)
                pil = Image.fromarray(np.ascontiguousarray(img))
                img = np.asarray(pil.filter(ImageFilter.GaussianBlur(sigma)))
            if self.jpeg_p > 0 and np.random.rand() < self.jpeg_p:
                q = int(np.random.randint(self.jpeg_qmin, self.jpeg_qmax + 1))
                img = jpeg_pil(img, q)
        return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))  # uint8 CHW
