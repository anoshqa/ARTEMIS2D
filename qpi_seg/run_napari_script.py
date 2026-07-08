import napari
import os
import tifffile
import numpy as np
import torch
from skimage.transform import resize


image_folder = r"C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\UNSEEN_MIP_1_REPEATED"
#import resized val images
output_mask_folder = r"C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\combined_mask_2"
#all val image files

corrected_mask_folder=r"C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\combined_corrected"


val_image_files=os.listdir(image_folder)

val_images_org=[tifffile.imread(os.path.join(image_folder, file)) for file in val_image_files]
val_image_stack = np.stack(val_images_org, axis=0)
mask_files=os.listdir(output_mask_folder)

masks_org=[tifffile.imread(os.path.join(output_mask_folder, file)).astype(np.uint16) for file in mask_files]
masks=[resize(mask, (418,418),order=0, anti_aliasing=False,preserve_range=True) for mask in masks_org]
print(masks[1].dtype)
mask_stack = np.stack(masks, axis=0)

print(f"val_image_stack shape: {val_image_stack.shape}")
print(f"mask_stack shape: {mask_stack.shape}")
out_file_name_masks_selected=[os.path.join(corrected_mask_folder, file) for file in mask_files]
# create the viewer and add the coins image
viewer = napari.Viewer()
viewer.add_image(val_image_stack, name='coins')
# add the labels
viewer.add_labels(mask_stack, name='segmentation')
#viewer = napari.Viewer()
napari.run()


for i in range(len(out_file_name_masks_selected)):
    image=val_images_org[i]
    edited_mask=mask_stack[i]
    w=image.shape[0]
    h=image.shape[1]
    mask_resized=resize(edited_mask, (w,h),order=0,anti_aliasing=False)
    tifffile.imwrite(
        out_file_name_masks_selected[i],
        mask_resized
    )