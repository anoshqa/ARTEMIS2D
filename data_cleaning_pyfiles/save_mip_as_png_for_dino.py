from skimage.transform import resize
import os
import matplotlib.pyplot as plt
import skimage
import numpy as np

mask_folder=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\phenotyping phase\Cell_mask_cleaned'
mask_files=sorted(os.listdir(mask_folder))
masks=[skimage.io.imread(os.path.join(mask_folder, maskfile)).astype(bool) for maskfile in sorted(os.listdir(mask_folder))]

mip_folder=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\phenotyping phase\MIP_proofread'
mips=[skimage.io.imread(os.path.join(mip_folder, mipfile)) for mipfile in sorted(os.listdir(mip_folder))]

masked_mips = [np.multiply(mip, mask) for mip, mask in zip(mips, masks)]

#mask_resized = [resize(image, (420, 420), order=0, anti_aliasing=False, preserve_range=True) for image in masks]


out_mask_folder=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\phenotyping phase\MIP_Cell_masked'

def save_gray_as_rgb_png(image,path,cmap='gray'):
    """Save a 2D image as an RGB PNG using a matplotlib colormap without altering pixel values."""
    img = np.asarray(image)
    plt.imsave(path, img, cmap=cmap, vmin=13330, vmax=14100)


png_paths = []
for idx, image in enumerate(masked_mips):
    name, _ = os.path.splitext(mask_files[idx])
    png_path = os.path.join(out_mask_folder, name + '_rgb.png')
    
    save_gray_as_rgb_png(image, png_path)