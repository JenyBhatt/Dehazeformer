import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import glob
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

from models import *
from models.dehazeformer import dehazeformer_b as Dehazeformer_B

# =========================
# SETTINGS
# =========================
INPUT_DIR  = "/content/Dehazeformer/data/test"
MODEL_PATH = "/content/Dehazeformer/pretrained_weights/Dehazing.pth"
OUTPUT_DIR = "/content/Dehazeformer/saved_models/indoor"
MODEL_NAME = "dehazeformer_b"

# ── TTA Config ──────────────────────────────────────────────
# Trained on 256×256 patches — keep patch_size=256 as base.
# Larger patch sizes will still work but use more VRAM.
TTA_CONFIGS = [
    # (patch_size, overlap, scale)
    (256, 0.50, 1.00),   # base config — matches training
    (256, 0.25, 1.00),   # different overlap
    (256, 0.50, 0.75),   # downscaled
    (256, 0.50, 1.25),   # upscaled
]

# ── Batch size for patch inference ──────────────────────────
# How many patches to run through the model at once.
# Increase if you have VRAM headroom, decrease if OOM.
PATCH_BATCH_SIZE = 24
# ────────────────────────────────────────────────────────────

# =========================
# Setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.backends.cudnn.benchmark     = True
torch.backends.cudnn.deterministic = False

print(f"🚀 Using device: {device}")
print(f"📐 TTA configs: {len(TTA_CONFIGS)} × 8 transforms = {len(TTA_CONFIGS) * 8} predictions per image")
print(f"⚡ Patch batch size: {PATCH_BATCH_SIZE}")


# =========================
# Load Model
# =========================
print("⚙️ Loading model weights...")
model = eval(MODEL_NAME)()
ckpt  = torch.load(MODEL_PATH, map_location="cpu")
sd    = ckpt.get('state_dict', ckpt)
sd    = OrderedDict({k[7:] if k.startswith('module.') else k: v for k, v in sd.items()})
model.load_state_dict(sd, strict=False)
model.to(device).eval()
print("✅ Model loaded!\n")


# =========================
# 8-Transform TTA Helpers
# =========================
def apply_tta(x):
    """Generate 8 geometric augmentations of x."""
    return [
        x,
        torch.flip(x, [3]),
        torch.flip(x, [2]),
        torch.flip(x, [2, 3]),
        torch.transpose(x, 2, 3),
        torch.flip(torch.transpose(x, 2, 3), [3]),
        torch.flip(torch.transpose(x, 2, 3), [2]),
        torch.flip(torch.transpose(x, 2, 3), [2, 3])
    ]

def inverse_tta(xs):
    """Reverse the 8 geometric augmentations."""
    return [
        xs[0],
        torch.flip(xs[1], [3]),
        torch.flip(xs[2], [2]),
        torch.flip(xs[3], [2, 3]),
        torch.transpose(xs[4], 2, 3),
        torch.transpose(torch.flip(xs[5], [3]), 2, 3),
        torch.transpose(torch.flip(xs[6], [2]), 2, 3),
        torch.transpose(torch.flip(xs[7], [2, 3]), 2, 3)
    ]


# =========================
# Gaussian Window
# =========================
_gaussian_cache = {}

def get_cached_gaussian(patch_size):
    if patch_size not in _gaussian_cache:
        half_w = patch_size / 2
        sigma  = 0.5 * half_w
        x      = np.arange(0, patch_size)
        y      = np.arange(0, patch_size)
        xx, yy = np.meshgrid(x, y)
        g      = np.exp(-((xx - half_w)**2 + (yy - half_w)**2) / (2 * sigma**2))
        _gaussian_cache[patch_size] = (
            torch.from_numpy(g).float().unsqueeze(0).unsqueeze(0).to(device)
        )
    return _gaussian_cache[patch_size]


# =========================
# Batched Grid Inference
# =========================
def grid_inference(model, img_tensor, patch_size, overlap):
    """
    Tiled inference with Gaussian blending.
    Patches are collected into batches of PATCH_BATCH_SIZE and
    processed together — much faster than one patch at a time.
    """
    b, c, h, w = img_tensor.shape
    stride = int(patch_size * (1 - overlap))

    pad_h = (stride - (h - patch_size) % stride) % stride if h > patch_size else max(0, patch_size - h)
    pad_w = (stride - (w - patch_size) % stride) % stride if w > patch_size else max(0, patch_size - w)

    padded = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    _, _, ph, pw = padded.shape

    output  = torch.zeros_like(padded)
    weights = torch.zeros_like(padded)
    gauss   = get_cached_gaussian(patch_size)  # (1,1,P,P)

    # ── Collect all patch positions ──
    positions = [
        (y, x)
        for y in range(0, ph - patch_size + 1, stride)
        for x in range(0, pw - patch_size + 1, stride)
    ]

    # ── Process in batches ──
    for batch_start in range(0, len(positions), PATCH_BATCH_SIZE):
        batch_pos = positions[batch_start : batch_start + PATCH_BATCH_SIZE]

        # Stack patches into a single batch tensor (N, C, P, P)
        patches = torch.cat(
            [padded[:, :, y:y+patch_size, x:x+patch_size] for y, x in batch_pos],
            dim=0
        )

        with torch.amp.autocast('cuda'):
            preds = torch.clamp(model(patches), 0, 1)  # (N, C, P, P)

        # Scatter predictions back with Gaussian weighting
        for idx, (y, x) in enumerate(batch_pos):
            pred = preds[idx:idx+1]   # (1, C, P, P)
            output [:, :, y:y+patch_size, x:x+patch_size] += pred  * gauss
            weights[:, :, y:y+patch_size, x:x+patch_size] += gauss

    return (output / weights.clamp(min=1e-6))[:, :, :h, :w]


# =========================
# Single TTA config
# =========================
def run_tta_config(model, img_tensor, patch_size, overlap, scale, orig_h, orig_w):
    if scale != 1.0:
        new_h = max(patch_size, int(orig_h * scale) // 8 * 8)
        new_w = max(patch_size, int(orig_w * scale) // 8 * 8)
        inp = F.interpolate(img_tensor, size=(new_h, new_w),
                            mode='bilinear', align_corners=False)
    else:
        inp = img_tensor

    aug_outputs = [
        grid_inference(model, aug, patch_size, overlap)
        for aug in apply_tta(inp)
    ]
    pred = torch.mean(torch.stack(inverse_tta(aug_outputs)), dim=0)

    if scale != 1.0:
        pred = F.interpolate(pred, size=(orig_h, orig_w),
                             mode='bilinear', align_corners=False)

    return pred.clamp(0, 1)


# =========================
# Inference Loop
# =========================
test_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.*')))
test_files = [f for f in test_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

to_tensor = transforms.ToTensor()
to_pil    = transforms.ToPILImage()

print(f"🚀 Running Advanced TTA on {len(test_files)} images...")
print(f"📂 Outputs will be saved to: {OUTPUT_DIR}\n")

with torch.no_grad():
    for file_path in tqdm(test_files, desc="🌙 Dehazing", unit="img"):
        filename    = os.path.basename(file_path)
        hazy_tensor = to_tensor(Image.open(file_path).convert('RGB')).unsqueeze(0).to(device)

        _, _, orig_h, orig_w = hazy_tensor.shape

        all_preds = []
        for patch_size, overlap, scale in TTA_CONFIGS:
            pred = run_tta_config(model, hazy_tensor, patch_size, overlap,
                                  scale, orig_h, orig_w)
            all_preds.append(pred)

        final_tensor = torch.mean(torch.stack(all_preds), dim=0).clamp(0, 1)
        to_pil(final_tensor.squeeze(0).cpu()).save(os.path.join(OUTPUT_DIR, filename))

print(f"\n✅ Done! {len(test_files)} dehazed images saved to:\n   {OUTPUT_DIR}")
