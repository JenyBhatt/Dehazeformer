#test 3

#1 Prepare environment and load model
import os
import torch

from models.dehazeformer import *
MODEL_NAME = "dehazeformer_b"
model = eval(MODEL_NAME)()

# CONFIGURATION

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

print("Model loaded!")

input_dir = '/content/Dehazeformer/data/test'
weights_path = '/content/Dehazeformer/weights/finetuned_phase3_highres_ema_24.39.pth'

output_dir = '/content/Dehazeformer/saved_models/indoor'
os.makedirs(output_dir, exist_ok=True)

print("Environment ready")

#3
import torch
from collections import OrderedDict
from models.dehazeformer import dehazeformer_b

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = dehazeformer_b()

ckpt = torch.load(weights_path, map_location=device)

state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

new_state_dict = OrderedDict()
for k,v in state_dict.items():
    name = k[7:] if k.startswith("module.") else k
    new_state_dict[name] = v

model.load_state_dict(new_state_dict, strict=False)

model = model.to(device)
model.eval()

print("Model loaded successfully")

#4 Grid inference
import torch.nn.functional as F

def grid_inference(model, img_tensor, patch_size=512, overlap=64):

    b, c, h, w = img_tensor.shape

    stride = patch_size - overlap

    output = torch.zeros((b, c, h, w)).to(img_tensor.device)
    weight = torch.zeros((b, c, h, w)).to(img_tensor.device)

    for y in range(0, h, stride):
        for x in range(0, w, stride):

            y1 = min(y + patch_size, h)
            x1 = min(x + patch_size, w)

            y0 = max(y1 - patch_size, 0)
            x0 = max(x1 - patch_size, 0)

            patch = img_tensor[:, :, y0:y1, x0:x1]

            with torch.no_grad():
                pred = torch.clamp(model(patch),0,1)

            output[:,:,y0:y1,x0:x1] += pred
            weight[:,:,y0:y1,x0:x1] += 1

    return output / weight

#5 TTA Inference

import glob
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

files = glob.glob(input_dir + "/*")

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

scales = [0.9, 1.0, 1.1]

for f in files:

    img = Image.open(f).convert("RGB")
    w,h = img.size

    preds = []

    for scale in scales:

        sw = int(w*scale)
        sh = int(h*scale)

        img_scaled = img.resize((sw,sh),Image.BICUBIC)

        tensor = to_tensor(img_scaled).unsqueeze(0).to(device)

        align = 16
        new_w = ((sw-1)//align+1)*align
        new_h = ((sh-1)//align+1)*align

        pad_w = new_w-sw
        pad_h = new_h-sh

        if pad_w!=0 or pad_h!=0:
            tensor = F.pad(tensor,(0,pad_w,0,pad_h),mode="reflect")

        aug_list = [
            tensor,
            torch.flip(tensor,[3]),
            torch.flip(tensor,[2]),
            torch.flip(tensor,[2,3]),
            torch.rot90(tensor,1,[2,3]),
            torch.flip(torch.rot90(tensor,1,[2,3]),[3]),
            torch.rot90(tensor,3,[2,3]),
            torch.flip(torch.rot90(tensor,3,[2,3]),[3]),
        ]

        for i, aug in enumerate(aug_list):

            out = grid_inference(model, aug)

            # reverse transforms
            if i==1:
                out = torch.flip(out,[3])
            elif i==2:
                out = torch.flip(out,[2])
            elif i==3:
                out = torch.flip(out,[2,3])
            elif i==4:
                out = torch.rot90(out,3,[2,3])
            elif i==5:
                out = torch.rot90(torch.flip(out,[3]),3,[2,3])
            elif i==6:
                out = torch.rot90(out,1,[2,3])
            elif i==7:
                out = torch.rot90(torch.flip(out,[3]),1,[2,3])

            if pad_w!=0 or pad_h!=0:
                out = out[:,:,:sh,:sw]

            out = F.interpolate(out,size=(h,w),mode="bilinear",align_corners=False)

            preds.append(out)

    final = torch.mean(torch.stack(preds),dim=0)

    result = to_pil(torch.clamp(final,0,1).squeeze().cpu())

    name = os.path.basename(f)
    result.save(os.path.join(output_dir,name))

print("24-TTA + Grid inference finished")


