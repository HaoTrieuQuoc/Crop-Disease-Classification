import torch
import numpy as np
from PIL import Image
from torchvision import transforms

IMG_SIZE = 224

rgb_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

def load_rgb(img):
    return rgb_tf(img).unsqueeze(0)

def load_spectral(img, ch):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img)

    if arr.ndim == 2:
        arr = np.repeat(arr[...,None], ch, axis=2)
    arr = arr[..., :ch]

    tensor = torch.tensor(arr).permute(2,0,1).float() / 255.0
    return tensor.unsqueeze(0)

def build_mask(use_rgb, use_ms, use_hs):
    return torch.tensor([[use_rgb, use_ms, use_hs]], dtype=torch.float32)
