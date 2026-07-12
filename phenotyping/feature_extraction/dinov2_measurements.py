import os
import gc
import numpy as np
import pandas as pd
import requests
import skimage
import torch
import matplotlib.pyplot as plt
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


def save_gray_as_rgb_png(image):
    """Save a 2D image as an RGB PNG using a matplotlib colormap without altering pixel values."""
    img = np.asarray(image)
    plt.imsave(img, cmap=cmap, vmin=0, vmax=4)


def prepare_image_for_dinov2(image):
    image_array = np.asarray(image)
    if image_array.ndim == 2:
        image_array = np.repeat(image_array[..., None], 3, axis=2)
    image_array = image_array.astype(np.float32)
    if image_array.max() > 1.0:
        image_array = image_array / 255.0
    return Image.fromarray(image_array)


#imports images
mask_folder=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\phenotyping phase\Mask_proofread'

masks=[skimage.io.imread(os.path.join(mask_folder, maskfile)) for maskfile in sorted(os.listdir(mask_folder))]
mask_resized = [resize(image, (420, 420), order=0, anti_aliasing=False, preserve_range=True) for image in masks]

png_paths = []
for idx, image in enumerate(mask_resized):
    png_path = os.path.join(os.getcwd(), f'tmp_mask_{idx}.png')
    save_gray_as_rgb_png(image, image, png_path)
    png_paths.append(png_path)
    
    
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
    for i in range(0, len(png_paths), batch_size):
        batch_images = png_paths[i:i+batch_size]

        # Stack batch
        batch_tensors = torch.stack([
            transform1(prepare_image_for_dinov2(Image.open(img_path).convert('RGB'))).to(dtype=torch.float32)
            for img_path in batch_images
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
featuresdf.to_csv('dino_features_masks.csv')