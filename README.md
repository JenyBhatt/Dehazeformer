# **Overview**

**DehazeFormer** is a transformer-based architecture for single image dehazing.

Here, we are incorporating the model, *Dehazeformer-B* ,a compressed version of the original Dehazeformer architecture.

### **Qualitative Results**
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

*Performance*
- Baseline inference: ~24.3 dB PSNR  
- TTA (test2.py): ~24.37 dB PSNR  

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
pip install -r requirements.txt

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

### Organize your images(Local/VsCode):
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
## Weights/Model
You can choose any of the weights as listed : 
- <a href="https://github.com/JenyBhatt/Dehazeformer/blob/main/pretrained_weights/finetuned_phase3_highres_ema_24.39.pth">**finetuned_phase3_highres_ema_24.39.pth**</a>
Model results
- <a href="https://github.com/JenyBhatt/Dehazeformer/blob/main/pretrained_weights/dehazing_best_model%20(4).pth">**finetuned_phase4_ssim.pth**</a>
Model results
 <br/> 

> Make sure to update the `--weights` path or the variable inside test scripts accordingly.
## 4. Run Inference
### Local/VsCode
```bash
python -m test --model dehazeformer-b --data_dir ./data --save_dir ./saved_models --dataset RESIDE-IN --exp indoor
```
### Google Colab
**TTA with other configs (24.368dB)** <br/>
```bash
!python test2.py
```

**Note:** The above configuration uses *test-time augmentation (TTA)*, which may increase inference time.

For faster inference, you may use the standard evaluation pipeline without TTA:<br/>
**Alternative Inference script:**
```bash
!python -m test \
    --model dehazeformer-b \
    --data_dir /content/Dehazeformer/data/test \
    --save_dir /content/Dehazeformer/saved_models \
    --dataset RESIDE-IN \
    --exp indoor \
    --weights /content/Dehazeformer/weights/finetuned_phase3_highres_ema_24.39.pth
```
You can download the pre-trained weights and select any of the mentioned <a href="https://github.com/JenyBhatt/Dehazeformer/tree/main/pretrained_weights">here</a>
<br/>
It is recommended to update the weights file path accordingly.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/github/license/JenyBhatt/Dehazeformer)

>**Note:** This repository uses the DehazeFormer-B model (a compressed variant) based on the original [DehazeFormer by IDKiro](https://github.com/IDKiro/DehazeFormer). It is a lightweight and computationally efficient variant of the original Dehazeformer, designed to reduce parameters and inference latency while maintaining competitive dehazing performance.
<br/>

```bibtex
@article{song2023vision,
  title={Vision Transformers for Single Image Dehazing},
  author={Song, Yuda and He, Zhuqing and Qian, Hui and Du, Xin},
  journal={IEEE Transactions on Image Processing},
  volume={32},
  pages={1927--1941},
  year={2023},
  publisher={IEEE}
}
