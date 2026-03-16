# **Overview**

**DehazeFormer** is a transformer-based architecture for single image dehazing, designed to:

-Handle high-resolution images via tiled inference

-Improve robustness via 8-fold geometric test-time augmentation (TTA)

-Produce visually clear and perceptually accurate results

-Easily integrate into research pipelines or automated image restoration tasks

Here, we are incorporating the model, *Dehazeformer-B* ,a compressed version of the original Dehazeformer architecture. The architecture is pictorially framed as below:
<br/>
<img src = "https://github.com/JenyBhatt/Dehazeformer/blob/main/images/architecture.jpeg" alt="architecture" width = "500">
<br/>

### **Sample Output**
<table>
  <tr>
    <td>
      <p align="center">Input</p>
      <img src="images/32_NTHazy.png" alt="Input Image" width="300"/>
    </td>
    <td>
      <p align="center">Output</p>
      <img src="images/sample_op.png" alt="Output Image" width="300"/>
    </td>
  </tr>
</table>

## 1. Installation
```bash
git clone https://github.com/JenyBhatt/Dehazeformer.git
cd Dehazeformer
```
## 2. Install Dependencies
### For Local / VSCode:
```bash
# PyTorch (GPU recommended, CUDA 12.1)
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Other required libraries
pip install --quiet opencv-python timm pytorch-msssim tqdm tensorboard tensorboardX
```
### For Google Colab:

#### Install PyTorch for Colab GPU runtime
```bash
!pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install --quiet opencv-python timm pytorch-msssim tqdm tensorboard tensorboardX
```

## 3. Dataset Preparation

### Organize your images(VsCode):
```bash
Dehazeformer/
├─ data/
│  └─ RESIDE-IN/
│      └─ indoor/       # input images here
├─ saved_models/
│  └─ indoor/           # outputs will be saved here
├─ models/
│  └─ dehazeformer.py
├─ utils/
│  └─ common.py, etc.
├─ test.py
```
### Organize your images(Google Colab):
```bash
Dehazeformer/
├─ data/
│  └─ test/           # Images to dehaze
├─ weights/
│  └─ finetuned_phase3_highres_ema_24.39.pth
```
## 4. Run Inference
### VsCode
```bash
python -m test --model dehazeformer-b --data_dir ./data --save_dir ./saved_models --dataset RESIDE-IN --exp indoor
```
### Google Colab
```bash
!python -m test \
    --model dehazeformer-b \
    --data_dir /content/Dehazeformer/data/test \
    --save_dir /content/Dehazeformer/saved_models \
    --dataset RESIDE-IN \
    --exp indoor \
    --weights /content/Dehazeformer/weights/finetuned_phase3_highres_ema_24.39.pth
```
**By running with TTA and other configs, we can slightly improve the psnr.**
**Alternative inference code for Colab :**
```bash
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

# Corrected import for Dehazeformer_B
from models.dehazeformer import dehazeformer_b as Dehazeformer_B

# =========================
# SETTINGS
# =========================

# ── TTA Config ──────────────────────────────────────────────
# Trained on 256×256 patches — keep patch_size=256 as base.
# Larger patch sizes will still work but use more VRAM.
TTA_CONFIGS = [
    # (patch_size, overlap, scale)
    (512, 0.50, 1.00),   # base config — matches training
    (512, 0.25, 1.00),   # different overlap
    (512, 0.50, 0.75),   # downscaled
    (512, 0.50, 1.25),   # upscaled
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

print(f"Using device: {device}")
print(f"TTA configs: {len(TTA_CONFIGS)} × 8 transforms = {len(TTA_CONFIGS) * 8} predictions per image")
print(f"Patch batch size: {PATCH_BATCH_SIZE}")


# =========================
# Load Model
# =========================
print("⚙️ Loading model weights...")
# Instantiate the model directly after explicit import
model = Dehazeformer_B()
ckpt  = torch.load(MODEL_PATH, map_location="cpu")
sd    = ckpt.get('state_dict', ckpt)
sd    = OrderedDict({k[7:] if k.startswith('module.') else k: v for k, v in sd.items()})
model.load_state_dict(sd, strict=False)
model.to(device).eval()
print("Model loaded!\n")


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

print(f"Running Advanced TTA on {len(test_files)} images...")
print(f"Outputs will be saved to: {OUTPUT_DIR}\n")

with torch.no_grad():
    for file_path in tqdm(test_files, desc="Dehazing", unit="img"):
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

print(f"\nDone. {len(test_files)} dehazed images saved to:\n   {OUTPUT_DIR}")
```
You can download the pre-trained weights <a href="https://github.com/JenyBhatt/Dehazeformer/blob/main/pretrained_weights/finetuned_phase3_highres_ema_24.39.pth">here</a>
## 5. Export / Download Outputs
### After inference, zip your results for download:
```bash
import shutil
shutil.make_archive("/content/dehazed_outputs", 'zip', "/content/Dehazeformer/saved_models/indoor")
```
The zip file dehazed_outputs.zip will contain all dehazed images.

### In Colab, download with:
```bash
from google.colab import files
files.download("/content/dehazed_outputs.zip")
```
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/github/license/JenyBhatt/Dehazeformer)
