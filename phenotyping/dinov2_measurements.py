import os
import requests
from PIL import Image
from torchvision import transforms
import torch
import gc
import pandas as pd
import skimage
from skimage.transform import resize
torch.cuda.empty_cache()
gc.collect()
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_vits14=dinov2_vits14.to(device).eval()
transform1 = transforms.Compose([  # Resize here
    transforms.ToTensor()
    #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       # std=[0.229, 0.224, 0.225])
])

#imports images
mask_folder=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\phenotyping phase\Mask_proofread'

masks=masks=[skimage.io.imread(os.path.join(mask_folder, maskfile) for maskfile in sorted(os.listdir(mask_folder)))]
mask_resized = [resize(image, (418, 418), order=0, anti_aliasing=False, preserve_range=True) for image in val_images_org]
patch_size = dinov2_vits14.patch_size # patchsize=14

#520//14
patch_h  = 112
patch_w  = 112
feat_dim = 384 # vits14
#feat_dim = 768 # vitb14
#feat_dim = 1024 # vitl14
# feat_dim = 1536 # vitg14
batch_size = 8  # T4 can handle this
total_features = []

batch_size = 8  # Adjust based on your GPU memory
total_features = []

with torch.no_grad():
    for i in range(0, len(masks), batch_size):
        batch_images = images[i:i+batch_size]

        # Stack batch
        batch_tensors = torch.stack([transform1(img) for img in batch_images])
        batch_tensors = batch_tensors.to(device)

        # Process batch
        features_dict = dinov2_vits14.forward_features(batch_tensors)
        features = features_dict['x_norm_clstoken']  # [batch_size, 768]

        # Move to CPU immediately
        total_features.append(features.cpu())

        # Clear memory
        del batch_tensors, features_dict, features
        torch.cuda.empty_cache()

        print(f"Processed {min(i+batch_size, len(images))}/{len(images)} images")

# Concatenate all on CPU
total_features = torch.cat(total_features, dim=0)
print(f"Final features shape: {total_features.shape}")
featuresdf=pd.DataFrame(total_features.numpy())
featuresdf.to_csv('dino_features.csv')