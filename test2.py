import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import glob

from models import *
from models.dehazeformer import dehazeformer_b as DehazeFormer_B

# =========================
# SETTINGS
# =========================
INPUT_DIR  = "/content/Dehazeformer/data/test"   # input images
MODEL_PATH = "/content/Dehazeformer/weights/Dehazing.pth"
OUTPUT_DIR = "./results_no_tta"
MODEL_NAME = "dehazeformer_b"

# =========================
# Setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f" Using device: {device}")

# =========================
# Load Model
# =========================
print("⚙️ Loading model...")
model = eval(MODEL_NAME)()

ckpt = torch.load(MODEL_PATH, map_location="cpu")
sd   = ckpt.get('state_dict', ckpt)

# Remove 'module.' if present
sd = {k.replace("module.", ""): v for k, v in sd.items()}

model.load_state_dict(sd, strict=False)
model.to(device).eval()

print(" Model loaded!\n")

# =========================
# Transforms
# =========================
to_tensor = transforms.ToTensor()
to_pil    = transforms.ToPILImage()

# =========================
# Inference Loop
# =========================
image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.*")))
image_paths = [p for p in image_paths if p.lower().endswith((".png", ".jpg", ".jpeg"))]

print(f" Found {len(image_paths)} images")

with torch.no_grad():
    for path in tqdm(image_paths, desc="Dehazing", unit="img"):
        img = Image.open(path).convert("RGB")
        tensor = to_tensor(img).unsqueeze(0).to(device)

        #  Simple forward pass (NO TTA)
        output = model(tensor)
        output = torch.clamp(output, 0, 1)

        # Save
        filename = os.path.basename(path)
        save_path = os.path.join(OUTPUT_DIR, filename)
        to_pil(output.squeeze(0).cpu()).save(save_path)

print(f"\n Done! Outputs saved to: {OUTPUT_DIR}")
