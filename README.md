# **Overview**

**DehazeFormer** is a transformer-based architecture for single image dehazing, designed to:

-Handle high-resolution images via tiled inference

-Improve robustness via 8-fold geometric test-time augmentation (TTA)

-Produce visually clear and perceptually accurate results

-Easily integrate into research pipelines or automated image restoration tasks

Here, we are incorporating the model, dehazeformer-b of the model sub-part.

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
