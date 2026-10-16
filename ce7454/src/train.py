#!/usr/bin/env python3
"""Train a model on the CE7454 challenge data.

Example:
  python train.py --model forensics --run-name forensics_r1 --epochs 30
"""
import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import CachedImageDataset, load_meta, stratified_split
from models import build_model, count_trainable, expand_to_split_bn, expand_to_grip_x4, set_bn_route

NORMS = {  # input normalization (mean, std) per original recipe
    "half": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1, 1]
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}
MEAN = torch.tensor(NORMS["half"][0]).view(1, 3, 1, 1)
STD = torch.tensor(NORMS["half"][1]).view(1, 3, 1, 1)


def set_norm(name):
    global MEAN, STD
    MEAN = torch.tensor(NORMS[name][0]).view(1, 3, 1, 1)
    STD = torch.tensor(NORMS[name][1]).view(1, 3, 1, 1)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resnet_features(m, x):
    """Global-average-pooled penultimate features of a torchvision ResNet (before the dropout/linear head)."""
    x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    x = m.layer4(m.layer3(m.layer2(m.layer1(x))))
    return torch.flatten(m.avgpool(x), 1)


def supcon_loss(z, labels, temp):
    """Supervised contrastive loss (Khosla et al., NeurIPS 2020), z L2-normalised (N, d)."""
    n = z.size(0)
    sim = z @ z.t() / temp
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, -1e9)
    pos = (labels[:, None] == labels[None, :]) & ~self_mask
    logprob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -((logprob * pos).sum(1) / pos.sum(1).clamp_min(1)).mean()


def normalize(x_uint8):
    x = x_uint8.float() / 255.0
    return (x - MEAN.to(x.device)) / STD.to(x.device)


class ModelEMA:
    """Exponential moving average of model parameters (buffers copied)."""

    def __init__(self, model, decay=0.999):
        import copy

        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for e, m in zip(self.module.parameters(), model.parameters()):
            e.lerp_(m.detach(), 1.0 - d)
        for e, m in zip(self.module.buffers(), model.buffers()):
            e.copy_(m)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    conf = np.zeros((3, 3), dtype=np.int64)
    for x, y in loader:
        x = normalize(x.to(device, non_blocking=True))
        y = y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        pred = logits.argmax(1)
        loss_sum += loss.item() * y.size(0)
        correct += (pred == y).sum().item()
        n += y.size(0)
        for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
            conf[t, p] += 1
    return loss_sum / n, correct / n, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        type=str,
        required=True,
    )
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--cache-dir", default="/disk/thuat/CE7454/cache")
    ap.add_argument("--train-split", default="train", help="cache split for training data")
    ap.add_argument("--out-root", default="/disk/thuat/CE7454/runs")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--val-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=None,
                    help="Optional independent loader/worker seed for matched architecture comparisons; default preserves old behavior")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--ema-decay", type=float, default=0.0, help="0 disables EMA")
    ap.add_argument("--jpeg-p", type=float, default=0.0)
    ap.add_argument("--jpeg-qmin", type=int, default=60)
    ap.add_argument("--jpeg-qmax", type=int, default=100)
    ap.add_argument("--blur-p", type=float, default=0.0)
    ap.add_argument("--blur-smax", type=float, default=1.0)
    ap.add_argument("--gray-p", type=float, default=0.0)
    ap.add_argument("--jitter-p", type=float, default=0.0)
    ap.add_argument("--jitter-s", type=float, default=0.1)
    ap.add_argument("--deterministic", action="store_true",
                    help="cudnn.benchmark off + deterministic algorithms (bit-reproducibility tests)")
    ap.add_argument("--flips", choices=["dihedral", "hflip", "rot90_hflip", "none"], default="dihedral",
                    help="train-time geometric augmentation (eval never flips)")
    # Original-recipe options (defaults reproduce the earlier experiments).
    ap.add_argument("--optim", choices=["adamw", "adam", "sgd"], default="adamw")
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    ap.add_argument("--sched", choices=["cosine", "step", "multistep", "plateau", "constant"], default="cosine")
    ap.add_argument("--anneal-epochs", type=int, default=0,
                    help="cosine only: anneal to --lr-floor over this many epochs, then hold the floor until --epochs (0 = off)")
    ap.add_argument("--lr-floor", type=float, default=0.0, help="floor as a fraction of the peak lr (with --anneal-epochs)")
    ap.add_argument("--milestones", default="0.5,0.75",
                    help="sched=multistep: fractions of the epoch budget where lr *= step-gamma")
    ap.add_argument("--step-every", type=int, default=10, help="sched=step: decay every N epochs")
    ap.add_argument("--step-gamma", type=float, default=0.9, help="sched=step: lr multiplier")
    ap.add_argument("--plateau-patience", type=int, default=5)
    ap.add_argument("--plateau-factor", type=float, default=0.1)
    ap.add_argument("--plateau-delta", type=float, default=0.001)
    ap.add_argument("--clip-grad", type=float, default=1.0, help="0 disables")
    ap.add_argument("--norm", choices=sorted(NORMS), default="half")
    ap.add_argument("--accum-steps", type=int, default=1,
                    help="gradient accumulation; loader batch = batch-size / accum-steps")
    ap.add_argument("--grip-blur-p", type=float, default=0.0)
    ap.add_argument("--grip-jpeg-p", type=float, default=0.0)
    ap.add_argument("--rrc-p", type=float, default=0.0)
    ap.add_argument("--chroma420-p", type=float, default=0.0, help="probability of chroma 4:2:0 subsampling augmentation (JPEG q100 4:2:0)")
    ap.add_argument("--crop-upscale", action="store_true", help="resize views smaller than --crop back up to it (large crops)")
    ap.add_argument("--hfgain-p", type=float, default=0.0, help="probability of high-frequency gain jitter (processed views)")
    ap.add_argument("--hfgain-max", type=float, default=1.6, help="HF gain g ~ log-uniform[1/max, max]")
    ap.add_argument("--flat-p", type=float, default=0.0, help="probability of flat-region crop sampling (training crops)")
    ap.add_argument("--flat-k", type=int, default=8, help="candidate crop positions for flat-region sampling")
    ap.add_argument("--patch-lambda", type=float, default=0.5,
                    help="models with a patch realness map (pi_net_h): weight of the per-location real-vs-synthetic BCE")
    ap.add_argument("--dense-lambda", type=float, default=0.5,
                    help="grip_nodown_dense only: weight of the per-location 3-class CE on the layer3 map")
    ap.add_argument("--rrc-size", type=int, default=256)
    ap.add_argument("--rrc-scale-min", type=float, default=0.08)
    ap.add_argument("--grip-blur-smax", type=float, default=3.0)
    ap.add_argument("--init-from", default=None,
                    help="checkpoint to initialise the weights from (fine-tuning / 'immunization', Yu et al. 2019)")
    ap.add_argument("--pcl-mode", default="none", choices=["none", "supcon", "ce"],
                    help="patch-level auxiliary loss on K random PxP patches of each training crop: supcon = DNA-Det "
                         "patchwise supervised contrastive (Khosla et al. 2020 loss, projection 512-512-128); "
                         "ce = matched control (cross-entropy on the identical patches). Kendall uncertainty weighting.")
    ap.add_argument("--pcl-k", type=int, default=4)
    ap.add_argument("--pcl-size", type=int, default=128)
    ap.add_argument("--pcl-temp", type=float, default=0.07)
    ap.add_argument("--pcl-lambda", type=float, default=1.0, help="fixed weight of the auxiliary loss (crop CE weight 1)")
    ap.add_argument("--match-p", type=float, default=0.0, help="train-side new-batch appearance matching probability")
    ap.add_argument("--match-mode", default="rgb", choices=["rgb", "luma"])
    ap.add_argument("--shuffle-p", type=float, default=0.0, help="patch-shuffle probability per view (Zheng et al. 2024)")
    ap.add_argument("--shuffle-tile", type=int, default=32)
    ap.add_argument("--rechain-p", type=float, default=0.0,
                    help="minimal re-encoding augmentation of the full image (near-identity crop-resize + JPEG Q85-95 4:2:0)")
    ap.add_argument("--desat-p", type=float, default=0.0, help="chroma reduction probability (processing view)")
    ap.add_argument("--desat-min", type=float, default=0.6, help="saturation factor ~ U[desat_min, 1]")
    ap.add_argument("--noise-p", type=float, default=0.0, help="additive Gaussian noise probability (processing view)")
    ap.add_argument("--noise-smax", type=float, default=4.0, help="noise sigma ~ U[0, smax] grey levels")
    ap.add_argument("--canon", default="none",
                    help="fixed input canonicalisation applied to every image before cropping, in training AND evaluation "
                         "(dataset.canonicalize): none | jpegQ | luma | lumajpegQ")
    ap.add_argument("--kd-teacher", default=None,
                    help="with --views>1: frozen teacher checkpoint (trained on the training split only) that sees the "
                         "geometric-only view 0; every student view is distilled to it (Hinton et al. 2015; clean-teacher "
                         "to degraded-student distillation as in ADD, AAAI 2022)")
    ap.add_argument("--kd-alpha", type=float, default=1.0)
    ap.add_argument("--kd-temp", type=float, default=4.0)
    ap.add_argument("--kd-views", choices=["all", "clean"], default="all",
                    help="all: distil every student view (0.5 KL_clean + 0.5 KL_processed); clean: only the clean view term, "
                         "same normalisation (0.5 KL_clean), the declared ablation")
    ap.add_argument("--supcon-views-lambda", type=float, default=0.0,
                    help=">0 with --views>1: add lambda * SupCon (Khosla et al. 2020, tau --pcl-temp) over the GAP features "
                         "of all views in the batch (positives = same class), projection 512-512-128, FP32")
    ap.add_argument("--views", type=int, default=1,
                    help=">1: AugMix-style consistency training with 1 clean + (views-1) processed views")
    ap.add_argument("--jsd-lambda", type=float, default=12.0, help="weight of the JSD consistency term")
    ap.add_argument("--sharpen-p", type=float, default=0.0, help="probability of sharpening before encoding")
    ap.add_argument("--soft-p", type=float, default=0.0, help="prob. of mild blur (sharpness jitter, before encoding)")
    ap.add_argument("--soft-smax", type=float, default=1.0)
    ap.add_argument("--sharpen-mode", choices=["composite", "unsharp"], default="composite")
    ap.add_argument("--social-mode", choices=["cf", "bfree"], default="cf",
                    help="processing chain used with --social-p: Community Forensics or B-Free social simulation")
    ap.add_argument("--social-region", action="store_true",
                    help="apply the CF chain to a random 1.45*crop region instead of the full image (faster)")
    ap.add_argument("--social-cf-interp", action="store_true",
                    help="RRC interpolation drawn from the 6 filters of the official CF code (default bilinear)")
    ap.add_argument("--social-p", type=float, default=0.0,
                    help="probability of the Community Forensics processing chain (resize/RRC/JPEG)")
    ap.add_argument("--grip-jpeg-qmin", type=int, default=30)
    ap.add_argument("--grip-jpeg-qmax", type=int, default=100)
    ap.add_argument("--frank-p", type=float, default=0.0)
    ap.add_argument("--he-scale", action="store_true")
    ap.add_argument("--pca-std", type=float, default=0.0)
    ap.add_argument("--weight-file", default=None,
                    help="JSON {filename: weight} for WeightedRandomSampler over training data")
    ap.add_argument("--sam-rho", type=float, default=0.0, help="SAM perturbation radius; 0 disables")
    ap.add_argument("--swad-start-frac", type=float, default=0.0,
                    help="SWAD (Cha et al., NeurIPS 2021; ImageNet variant): from this fraction of the optimizer steps the "
                         "LR is held constant and the weights are averaged after every step; then BN statistics are "
                         "recomputed on training data and the averaged model is saved as swad.pt. 0 disables")
    ap.add_argument("--swad-bn-batches", type=int, default=200)
    ap.add_argument("--swa-start-frac", type=float, default=0.0,
                    help="fraction of schedule after which to average weights; 0 disables")
    args = ap.parse_args()

    seed_all(args.seed)
    set_norm(args.norm)
    assert args.batch_size % args.accum_steps == 0
    assert not (args.sam_rho > 0 and args.accum_steps > 1), "SAM with accumulation not supported"
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before the first cuBLAS call
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
    device = "cuda"
    out_dir = os.path.join(args.out_root, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "args.json"), "w") as fp:
        json.dump(vars(args), fp, indent=2)

    meta = load_meta(args.cache_dir, args.train_split)
    labels = np.asarray(meta["labels"])
    if args.train_split != "train" and args.val_frac > 0:
        # Combined caches (train + pseudo-labelled images) start with the 15,000 training images in
        # the original order: keep exactly the same held-out val split, and train on the rest of the
        # training images plus every appended image.
        base = np.asarray(load_meta(args.cache_dir, "train")["labels"])
        assert np.array_equal(labels[:len(base)], base), "combined cache must start with train"
        tr0, va_idx = stratified_split(base, args.val_frac, args.split_seed)
        tr_idx = np.concatenate([tr0, np.arange(len(base), len(labels))])
    else:
        tr_idx, va_idx = stratified_split(labels, args.val_frac, args.split_seed)
    print(f"split: {len(tr_idx)} train / {len(va_idx)} val")

    train_ds = CachedImageDataset(
        args.cache_dir, args.train_split, tr_idx, labels[tr_idx], train=True, crop=args.crop,
        jpeg_p=args.jpeg_p, jpeg_qmin=args.jpeg_qmin, jpeg_qmax=args.jpeg_qmax,
        blur_p=args.blur_p, blur_smax=args.blur_smax,
        gray_p=args.gray_p, jitter_p=args.jitter_p, jitter_s=args.jitter_s, flips=args.flips,
        grip_blur_p=args.grip_blur_p, grip_jpeg_p=args.grip_jpeg_p, rrc_p=args.rrc_p,
        rrc_size=args.rrc_size, frank_p=args.frank_p, he_scale=args.he_scale, pca_std=args.pca_std,
        grip_blur_smax=args.grip_blur_smax, grip_jpeg_qmin=args.grip_jpeg_qmin,
        grip_jpeg_qmax=args.grip_jpeg_qmax, rrc_scale_min=args.rrc_scale_min, social_p=args.social_p,
        views=args.views, social_cf_interp=args.social_cf_interp, social_region=args.social_region, social_mode=args.social_mode,
        sharpen_p=args.sharpen_p, sharpen_mode=args.sharpen_mode, soft_p=args.soft_p, soft_smax=args.soft_smax,
        canon=args.canon, noise_p=args.noise_p, noise_smax=args.noise_smax, desat_p=args.desat_p, desat_min=args.desat_min, rechain_p=args.rechain_p, match_p=args.match_p, match_mode=args.match_mode, shuffle_p=args.shuffle_p, shuffle_tile=args.shuffle_tile, flat_p=args.flat_p, flat_k=args.flat_k, hfgain_p=args.hfgain_p, hfgain_max=args.hfgain_max, crop_upscale=args.crop_upscale, chroma420_p=args.chroma420_p,
    )
    val_ds = None
    if len(va_idx) > 0:
        val_ds = CachedImageDataset(args.cache_dir, args.train_split, va_idx, labels[va_idx], train=False, canon=args.canon)
    sampler = None
    if args.weight_file:
        wmap = json.load(open(args.weight_file))
        fnames = meta["filenames"]
        weights = torch.tensor([wmap.get(fnames[i], 1.0) for i in tr_idx], dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(tr_idx),
                                                         replacement=True)
        print(f"weighted sampling from {args.weight_file}: mean {weights.mean():.2f}")
    train_ld = DataLoader(
        train_ds, batch_size=args.batch_size // args.accum_steps, shuffle=(sampler is None), sampler=sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=True,
        generator=(None if args.data_seed is None else torch.Generator().manual_seed(args.data_seed)),
    )
    val_ld = None
    if val_ds is not None:
        val_ld = DataLoader(
            val_ds, batch_size=args.val_batch_size, shuffle=False, num_workers=4, pin_memory=True,
        )

    if args.model == "grip_ycc":
        assert args.norm == "imagenet", "YCbCrStem expects ImageNet-normalised input"
    if args.model.endswith("_sbn"):
        assert args.views > 1, "split BN needs clean + processed views (--views 2)"
    model = build_model(args.model).to(device).to(memory_format=torch.channels_last)
    if args.init_from:
        init = torch.load(args.init_from, map_location="cpu", weights_only=False)
        if args.model == "grip_nodown_x4":
            if init.get("args", {}).get("model") not in ("grip_nodown", "grip_nodown_x4"):
                raise ValueError("grip_nodown_x4 initialization requires a plain GRIP or x4 checkpoint")
            model.load_state_dict(expand_to_grip_x4(model, init["model"]))
        elif args.model == "grip_nodown_dense" and "dense_head.weight" not in init["model"]:
            # plain GRIP checkpoint -> dense-auxiliary model: new auxiliary head at its fresh init (same inference function)
            sd = dict(init["model"])
            sd.update({k: v for k, v in model.state_dict().items() if k.startswith("dense_head")})
            model.load_state_dict(sd)
        elif (args.model.endswith("_pvtw") or args.model.endswith("_pvmed")) and \
                init.get("args", {}).get("model") == args.model.rsplit("_", 1)[0]:
            # base checkpoint -> + patch-vote head (fresh) with scale 0: identical function at initialisation
            sd = dict(init["model"])
            sd.update({k: v for k, v in model.state_dict().items() if k.startswith("patch_")})
            model.load_state_dict(sd)
        elif args.model.endswith("_twcov") and init.get("args", {}).get("model") in (args.model[:-3], args.model[:-6]):
            # <base>_tw checkpoint -> + flat-weighted covariance branch; head = old head (first 512 cols) + zero cov cols,
            # so the function is unchanged at initialisation
            sd = dict(init["model"])
            own = model.state_dict()
            w = own["fc.1.weight"].clone(); w[:, :512] = sd["fc.1.weight"]; sd["fc.1.weight"] = w
            sd.update({k: v for k, v in own.items() if k.startswith("cov_")})
            model.load_state_dict(sd)
        elif args.model.endswith("_tw") and init.get("args", {}).get("model") == args.model[:-3]:
            # same backbone/head, only the pooling changes (parameter-free): load the base checkpoint as is
            model.load_state_dict(init["model"])
        elif args.model in ("grip_nodown_ibna", "grip_nodown_ibnb3") and "conv1.weight" in init["model"] \
                and init.get("args", {}).get("model") == "grip_nodown":
            # plain GRIP checkpoint -> IBN-a / IBN-b3. IBN-a: BN affine of each converted block's bn1 is split into the IN half
            # (affine copied, stats dropped) and the BN half (affine + running stats copied). IBN-b3: as IBN-b plus a fresh IN.
            own = model.state_dict()
            sd = {}
            for k, v in init["model"].items():
                if k in own:
                    sd[k] = v
                    continue
                for suf in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                    base = k[: -len(suf) - 1] if k.endswith("." + suf) else None
                    if base is not None and base + ".IN." + suf in own or (base is not None and base + ".BN." + suf in own):
                        h = own[base + ".BN.weight"].shape[0] if base + ".BN.weight" in own else 0
                        n_in = v.shape[0] - h if v.dim() else 0
                        if suf in ("weight", "bias") and base + ".IN." + suf in own:
                            sd[base + ".IN." + suf] = v[:n_in]
                        if base + ".BN." + suf in own:
                            sd[base + ".BN." + suf] = v if suf == "num_batches_tracked" else v[n_in:]
            sd = {k: v for k, v in sd.items() if not (args.model == "grip_nodown_ibnb3" and k.startswith("bn1.running"))}
            sd.pop("bn1.num_batches_tracked", None) if args.model == "grip_nodown_ibnb3" else None
            missing = [k for k in own if k not in sd]
            for k in missing:
                sd[k] = own[k]
            print(f"  IBN init: {len(missing)} keys at fresh init: {missing}", flush=True)
            model.load_state_dict(sd)
        elif args.model == "grip_nodown_ibnb" and "layer1.1.ibn.weight" not in init["model"]:
            # plain GRIP checkpoint -> IBN-b: stem BN affine -> stem IN affine (running stats dropped), new IN layers at identity affine
            sd = {k: v for k, v in init["model"].items() if k not in ("bn1.running_mean", "bn1.running_var", "bn1.num_batches_tracked")}
            sd.update({k: v for k, v in model.state_dict().items() if ".ibn." in k})
            model.load_state_dict(sd)
        elif args.model in ("grip_nodown_rw", "grip_nodown_rwy") and "conv1.weight" in init["model"]:
            # plain GRIP checkpoint -> whitening model: same conv1 weights behind the parameter-free RadialWhiten
            sd = {("conv1.1.weight" if k == "conv1.weight" else k): v for k, v in init["model"].items()}
            sd.update({k: v for k, v in model.state_dict().items() if k.startswith("conv1.0.")})
            model.load_state_dict(sd)
        else:
            model.load_state_dict(expand_to_split_bn(model, init["model"]) if args.model.endswith("_sbn") else init["model"])
        print(f"initialised from {args.init_from} (epoch {init.get('epoch')})", flush=True)
    n_params = count_trainable(model)
    print(f"model={args.model} trainable_params={n_params:,} (limit 12,300,000)")
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None

    teacher = None
    if args.kd_teacher:
        assert args.views > 1, "--kd-teacher needs --views 2 (teacher sees the geometric-only view 0)"
        tck = torch.load(args.kd_teacher, map_location="cpu", weights_only=False)
        assert tck["args"].get("norm", "half") == args.norm, "teacher and student must share input normalisation"
        assert tck["args"].get("train_split", "train") == "train" and tck["args"].get("val_frac") == args.val_frac \
            and tck["args"].get("split_seed") == args.split_seed, "teacher must be trained on the same training split"
        with torch.random.fork_rng(devices=[]):
            teacher = build_model(tck["args"]["model"]).to(device).to(memory_format=torch.channels_last)
        teacher.load_state_dict(tck["model"])
        teacher.eval()
        for p_ in teacher.parameters():
            p_.requires_grad_(False)
        print(f"KD teacher {args.kd_teacher} ({tck['args']['model']}, epoch {tck.get('epoch')}), "
              f"alpha {args.kd_alpha}, T {args.kd_temp}", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    pcl_head = None
    if args.pcl_mode != "none" or args.supcon_views_lambda > 0:
        assert args.model in ("grip_nodown", "baseline", "resnet18_nodown") and args.sam_rho == 0
        assert args.pcl_mode == "none" or args.views == 1
        assert args.supcon_views_lambda == 0 or (args.views > 1 and args.jsd_lambda == 0)
        assert args.pcl_k >= 1 and 32 <= args.pcl_size <= args.crop and args.pcl_temp > 0
        # training-only projection head (not part of the deployed model / its state dict)
        with torch.random.fork_rng(devices=[]):   # keep the CPU RNG (data order) identical to the no-head baseline
            pcl_head = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, 128)).to(device)
        if args.pcl_mode != "ce":   # the CE control does not use the head
            params = params + list(pcl_head.parameters())
        n_aux = sum(p.numel() for p in pcl_head.parameters()) if args.pcl_mode != "ce" else 0
        print(f"training-only projection head: {n_aux:,} params -> training total {n_params + n_aux:,}, "
              f"deployed {n_params:,}", flush=True)
        assert n_params + n_aux <= 12_300_000, "training total exceeds the parameter cap"
    n_params_training = n_params + (n_aux if pcl_head is not None else 0)
    if args.optim == "adamw":
        # No weight decay on norm layers and biases.
        decay = [p for p in params if p.ndim > 1]
        no_decay = [p for p in params if p.ndim <= 1]
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": args.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps,
        )
    elif args.optim == "adam":  # original recipes: coupled L2, usually 0
        opt = torch.optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps,
                               weight_decay=args.weight_decay)
    else:
        opt = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)
    steps_per_epoch = len(train_ld) // args.accum_steps  # optimizer steps
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    swad_start_step = int(args.swad_start_frac * total_steps)
    swad_state, swad_n = None, 0

    anneal_steps = args.anneal_epochs * steps_per_epoch

    def lr_at(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if args.sched != "cosine":
            return 1.0
        if args.anneal_epochs > 0:
            # cosine from 1 to lr_floor over the first anneal_epochs, then hold lr_floor until --epochs (every checkpoint
            # after the anneal is a finished model; training can be stopped at any time and the tail selected on val/dev)
            if step >= anneal_steps:
                return args.lr_floor
            t = (step - warmup_steps) / max(1, anneal_steps - warmup_steps)
            return args.lr_floor + (1 - args.lr_floor) * 0.5 * (1 + np.cos(np.pi * t))
        if args.swad_start_frac > 0:
            step = min(step, swad_start_step)   # SWAD: constant LR from the averaging start
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * t))

    # Per-step factor (warmup / cosine); "step" and "plateau" decays are applied per epoch
    # on top of it through `epoch_mult`.
    epoch_mult = 1.0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_at(s) * epoch_mult)
    plateau_best, plateau_bad = -1.0, 0

    log_path = os.path.join(out_dir, "log.csv")
    with open(log_path, "w", newline="") as fp:
        csv.writer(fp).writerow(
            ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "sec"]
        )

    best_acc = 0.0
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum, correct, n = 0.0, 0, 0
        aux_sum, ce_sum = 0.0, 0.0
        opt.zero_grad(set_to_none=True)
        for it, (x, y) in enumerate(train_ld):
            x = normalize(x.to(device, non_blocking=True))
            if args.views == 1:
                x = x.to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if args.views > 1:
                    # x: (B, V, C, H, W). CE on every view + JSD between the views' predictions
                    # (Hendrycks et al., AugMix, ICLR 2020).
                    B, V = x.shape[:2]
                    if args.model.endswith("_sbn"):
                        # P1 split BN: clean view 0 through BN set 0, processed views through BN set 1
                        # (separate forwards so that each set only sees its own domain's batch statistics).
                        set_bn_route(model, 0)
                        l0 = model(x[:, 0].contiguous(memory_format=torch.channels_last)).float()
                        set_bn_route(model, 1)
                        l1 = model(x[:, 1:].flatten(0, 1).contiguous(memory_format=torch.channels_last)).float()
                        set_bn_route(model, 0)
                        lv = torch.cat([l0.view(B, 1, -1), l1.view(B, V - 1, -1)], 1)
                    elif args.supcon_views_lambda > 0:
                        # one backbone pass: GAP features -> the model's own head (== model(x)) and the projection
                        fv = resnet_features(model, x.flatten(0, 1).contiguous(memory_format=torch.channels_last))
                        lv = model.fc(fv).float().view(B, V, -1)
                        with torch.autocast("cuda", enabled=False):
                            z = F.normalize(pcl_head(fv.float()), dim=1)
                            sc = supcon_loss(z, y.repeat_interleave(V), args.pcl_temp)
                    else:
                        lv = model(x.flatten(0, 1).contiguous(memory_format=torch.channels_last)).float().view(B, V, -1)
                    loss = F.cross_entropy(lv.flatten(0, 1), y.repeat_interleave(V),
                                           label_smoothing=args.label_smoothing)
                    if hasattr(model, "patch_real") and args.patch_lambda > 0:
                        pr_ = model.patch_real_map.float()[:, 0]                # (B*V, h, w)
                        tr_ = (y.repeat_interleave(V) == 0).float()[:, None, None].expand_as(pr_)
                        loss = loss + args.patch_lambda * F.binary_cross_entropy_with_logits(pr_, tr_)
                    if hasattr(model, "dense_head") and args.dense_lambda > 0:
                        dl_ = model.dense_logits.float()                        # (B*V, 3, h, w)
                        t_ = y.repeat_interleave(V)[:, None, None].expand(-1, dl_.shape[2], dl_.shape[3])
                        loss = loss + args.dense_lambda * F.cross_entropy(dl_, t_)
                    pv = F.softmax(lv, dim=-1)
                    pm = pv.mean(1, keepdim=True).clamp(1e-7, 1).log()
                    jsd = (pv * (pv.clamp(1e-7, 1).log() - pm)).sum(-1).mean()
                    loss = loss + args.jsd_lambda * jsd
                    if teacher is not None:
                        with torch.no_grad():
                            tl = teacher(x[:, 0].contiguous(memory_format=torch.channels_last)).float()
                        T = args.kd_temp
                        pt = F.softmax(tl / T, dim=-1)[:, None, :].expand(-1, V, -1)
                        kdv = F.kl_div(F.log_softmax(lv / T, dim=-1), pt, reduction="none").sum(-1)   # (B, V)
                        if args.kd_views == "clean":
                            kdv = kdv * (torch.arange(V, device=kdv.device) == 0).float()[None, :]
                        kd = kdv.mean() * T * T
                        ce_sum += loss.item() * y.size(0)
                        aux_sum += kd.item() * y.size(0)
                        loss = loss + args.kd_alpha * kd
                    if args.supcon_views_lambda > 0:
                        ce_sum += loss.item() * y.size(0)
                        aux_sum += sc.item() * y.size(0)
                        loss = loss + args.supcon_views_lambda * sc
                    logits = lv[:, 0]
                else:
                    logits = model(x)
                    loss = F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)
                    if hasattr(model, "patch_real") and args.patch_lambda > 0:
                        pr_ = model.patch_real_map.float()[:, 0]
                        loss = loss + args.patch_lambda * F.binary_cross_entropy_with_logits(pr_, (y == 0).float()[:, None, None].expand_as(pr_))
                    if hasattr(model, "dense_head") and args.dense_lambda > 0:
                        dl_ = model.dense_logits.float()
                        t_ = y[:, None, None].expand(-1, dl_.shape[2], dl_.shape[3])
                        loss = loss + args.dense_lambda * F.cross_entropy(dl_, t_)
                    if args.pcl_mode != "none":
                        Bp, _, Hp, Wp = x.shape
                        P, K = args.pcl_size, args.pcl_k
                        oy = torch.randint(0, Hp - P + 1, (Bp, K)).tolist()
                        ox = torch.randint(0, Wp - P + 1, (Bp, K)).tolist()
                        patches = torch.stack([x[b_, :, oy[b_][k_]:oy[b_][k_] + P, ox[b_][k_]:ox[b_][k_] + P]
                                               for b_ in range(Bp) for k_ in range(K)]).contiguous(memory_format=torch.channels_last)
                        yp = y.repeat_interleave(K)
                        if args.pcl_mode == "supcon":
                            fp = resnet_features(model, patches)
                            with torch.autocast("cuda", enabled=False):
                                aux = supcon_loss(F.normalize(pcl_head(fp.float()), dim=1), yp, args.pcl_temp)
                        else:
                            aux = F.cross_entropy(model(patches).float(), yp)
                        ce_sum += loss.item() * y.size(0)
                        aux_sum += aux.item() * y.size(0)
                        loss = loss + args.pcl_lambda * aux
            (loss / args.accum_steps).backward()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
            if (it + 1) % args.accum_steps:
                continue
            if args.sam_rho > 0:
                # SAM (Foret et al. 2021): ascend to the worst-case nearby
                # weights, take the real gradient there, descend from the
                # original point.
                with torch.no_grad():
                    gnorm = torch.norm(torch.stack([
                        p.grad.norm() for p in model.parameters() if p.grad is not None
                    ])) + 1e-12
                    eps = {}
                    for p in model.parameters():
                        if p.grad is not None:
                            e = p.grad * (args.sam_rho / gnorm)
                            p.add_(e)
                            eps[p] = e
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss2 = F.cross_entropy(model(x), y, label_smoothing=args.label_smoothing)
                loss2.backward()
                with torch.no_grad():
                    for p, e in eps.items():
                        p.sub_(e)
            if args.clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            if ema is not None:
                ema.update(model)
            step += 1
            if args.swad_start_frac > 0 and step > swad_start_step:
                with torch.no_grad():
                    if swad_state is None:
                        swad_state = {k: v.detach().clone().float() for k, v in model.state_dict().items()
                                      if v.dtype.is_floating_point}
                        swad_n = 1
                    else:
                        swad_n += 1
                        for k, v in model.state_dict().items():
                            if k in swad_state:
                                swad_state[k] += (v.detach().float() - swad_state[k]) / swad_n
        train_loss, train_acc = loss_sum / n, correct / n

        if val_ld is None:
            val_loss, val_acc = 0.0, 0.0
            conf = np.zeros((3, 3), dtype=np.int64)
        else:
            val_loss, val_acc, conf = evaluate(model, val_ld, device)
        ema_line = ""
        ema_won = False
        if ema is not None and val_ld is not None:
            ema_loss, ema_acc, ema_conf = evaluate(ema.module, val_ld, device)
            ema_line = f" | ema {ema_loss:.4f}/{ema_acc:.4f}"
            if ema_acc > val_acc:
                val_acc, val_loss, conf = ema_acc, ema_loss, ema_conf
                ema_won = True
        dt = time.time() - t0
        lr_now = opt.param_groups[0]["lr"]
        print(
            f"epoch {epoch:3d} | lr {lr_now:.2e} | train {train_loss:.4f}/{train_acc:.4f} "
            f"| val {val_loss:.4f}/{val_acc:.4f}{ema_line} | {dt:.0f}s",
            flush=True,
        )
        if pcl_head is not None or teacher is not None:
            print(f"  aux terms: crop CE {ce_sum / max(n, 1):.4f} | auxiliary ({'kd' if teacher is not None else (args.pcl_mode if args.pcl_mode != 'none' else 'supcon-views')}) "
                  f"{aux_sum / max(n, 1):.4f}", flush=True)
        with open(log_path, "a", newline="") as fp:
            csv.writer(fp).writerow(
                [epoch, f"{lr_now:.6e}", f"{train_loss:.6f}", f"{train_acc:.6f}",
                 f"{val_loss:.6f}", f"{val_acc:.6f}", f"{dt:.1f}"]
            )

        # Epoch-level lr decays of the original recipes.
        old_mult = epoch_mult
        if args.sched == "multistep" and epoch in {int(round(float(f) * args.epochs)) for f in args.milestones.split(",")}:
            epoch_mult *= args.step_gamma  # e.g. /10 at 50% and 75% of the budget (He et al. 2016 style)
        elif args.sched == "step" and epoch % args.step_every == 0:
            epoch_mult *= args.step_gamma  # NPR / FreqNet: lr *= gamma every N epochs
        elif args.sched == "plateau" and val_ld is not None:
            # GRIP training code: early-stopping counter on val accuracy; when it runs out,
            # lr /= 10 and the counter restarts from the current score (we keep training
            # for the full epoch budget instead of stopping once lr < 1e-6).
            if val_acc > plateau_best + args.plateau_delta:
                plateau_best, plateau_bad = val_acc, 0
            else:
                plateau_bad += 1
                if plateau_bad >= args.plateau_patience:
                    epoch_mult *= args.plateau_factor
                    plateau_best, plateau_bad = val_acc, 0
                    print(f"  plateau: lr x{args.plateau_factor}", flush=True)
        if epoch_mult != old_mult:
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * lr_at(step) * epoch_mult

        if args.swa_start_frac > 0 and epoch >= int(args.swa_start_frac * args.epochs):
            if "swa_state" not in dir():
                import copy as _copy
                swa_state = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
                swa_n = 1
            else:
                swa_n += 1
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        swa_state[k] += (v.detach().float() - swa_state[k]) / swa_n
                    else:
                        swa_state[k] = v.detach().clone()
        ckpt = {
            "model": (ema.module if ema_won else model).state_dict(),
            "from_ema": ema_won, "epoch": epoch, "val_acc": val_acc,
            "args": vars(args), "n_params": n_params, "n_params_training": n_params_training,
            "confusion": conf.tolist(),
        }
        if pcl_head is not None:
            ckpt["pcl_head"] = pcl_head.state_dict()   # training-only; inference uses ckpt["model"] alone
        torch.save(ckpt, os.path.join(out_dir, "last.pt"))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(ckpt, os.path.join(out_dir, "best.pt"))
            print(f"  new best: {best_acc:.4f}\n  confusion (rows=true 0/1/2):\n{conf}", flush=True)

    if args.swad_start_frac > 0 and swad_state is not None:
        sd = model.state_dict()
        model.load_state_dict({k: (swad_state[k].to(sd[k].dtype) if k in swad_state else sd[k]) for k in sd})
        # recompute BN statistics of the averaged weights on training data (cumulative average)
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.reset_running_stats()
                m.momentum = None
        model.train()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i, (x, y) in enumerate(train_ld):
                if i >= args.swad_bn_batches:
                    break
                x = normalize(x.to(device))
                if x.dim() == 5:
                    x = x.flatten(0, 1)
                model(x.contiguous(memory_format=torch.channels_last))
        model.eval()
        swad_acc = 0.0
        if val_ld is not None:
            _, swad_acc, _ = evaluate(model, val_ld, device)
        print(f"swad: averaged {swad_n} iterates from step {swad_start_step}; val acc {swad_acc:.4f}", flush=True)
        torch.save({"model": model.state_dict(), "epoch": args.epochs, "val_acc": swad_acc, "args": vars(args),
                    "n_params": n_params, "from_swad": True, "swad_n": swad_n}, os.path.join(out_dir, "swad.pt"))
    if args.swa_start_frac > 0 and "swa_state" in dir():
        model.load_state_dict({k: v.to(dtype=dict(model.state_dict())[k].dtype)
                               for k, v in swa_state.items()})
        # Recalibrate BN running stats for the averaged weights.
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
                m.momentum = None
        model.train()
        with torch.no_grad():
            for i, (x, y) in enumerate(train_ld):
                if i >= 50:
                    break
                x = normalize(x.to(device)).to(memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    model(x)
        model.eval()
        swa_acc = 0.0
        if val_ld is not None:
            _, swa_acc, _ = evaluate(model, val_ld, device)
            print(f"swa val acc: {swa_acc:.4f}")
        torch.save({"model": model.state_dict(), "epoch": args.epochs, "val_acc": swa_acc,
                    "args": vars(args), "n_params": n_params, "from_swa": True},
                   os.path.join(out_dir, "swa.pt"))
    print(f"done. best val acc {best_acc:.4f} -> {out_dir}/best.pt")


if __name__ == "__main__":
    main()
