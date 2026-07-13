import os
import gc
import numpy as np
import pandas as pd
import requests
import skimage
import torch
from PIL import Image
from skimage.transform import resize
from torchvision import transforms

torch.cuda.empty_cache()
gc.collect()
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_vits14 = dinov2_vits14.to(device).eval()
transform1 = transforms.Compose([
    transforms.ToTensor(),
])


filenames = []
folder_path = r"phenotyping\feature_extraction\Mask_resized_dino" 
masks=[]
with torch.no_grad():
  for img_name in sorted(os.listdir(folder_path)):
    img_path = os.path.join(folder_path, img_name)
    img = Image.open(img_path).convert('RGB')
    masks.append(img)
    filenames.append(img_name)

    
patch_size = dinov2_vits14.patch_size # patchsize=14

#520//14
patch_h  = 30
patch_w  = 30
feat_dim = 384 # vits14
#feat_dim = 768 # vitb14
#feat_dim = 1024 # vitl14
# feat_dim = 1536 # vitg14
total_features = []

batch_size = 4  
total_features = []

with torch.no_grad():
    for i in range(0, len(masks), batch_size):
        batch_masks = masks[i:i+batch_size]

        # Stack batch
        batch_tensors = torch.stack([
            transform1(mask).to(dtype=torch.float32)
            for mask in batch_masks
        ])
        batch_tensors = batch_tensors.to(device=device, dtype=torch.float32)

        # Process batch
        features_dict = dinov2_vits14.forward_features(batch_tensors)
        features = features_dict['x_norm_clstoken']  # [batch_size, 384]

        # Move to CPU immediately
        total_features.append(features.cpu())

        # Clear memory
        del batch_tensors, features_dict, features
        torch.cuda.empty_cache()

        print(f"Processed {min(i+batch_size, len(masks))}/{len(masks)} images")

# Concatenate all on CPU
total_features = torch.cat(total_features, dim=0)
print(f"Final features shape: {total_features.shape}")
featuresdf=pd.DataFrame(total_features.numpy())
featuresdf.insert(0, 'mask_file', filenames)
featuresdf.to_csv('dino_features_masks.csv', index=False)