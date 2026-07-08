import napari
import os
import tifffile
import numpy as np
import torch
from skimage.transform import resize


image_folder = r"D:\TRAINING_DATA_FINAL\TEST_MIP_REPEATED"
#import resized val images
output_mask_folder = r"D:\TRAINING_DATA_FINAL\COMBINED_TEST_MIP"
#all val image files

corrected_mask_folder=r"D:\TRAINING_DATA_FINAL\COMBINED_TEST_MIP_CORRECT"


val_image_files=os.listdir(image_folder)

val_images_org=[tifffile.imread(os.path.join(image_folder, file)) for file in val_image_files]
val_image_stack = np.stack(val_images_org, axis=0)
mask_files=os.listdir(output_mask_folder)

masks_org=[tifffile.imread(os.path.join(output_mask_folder, file)) for file in mask_files]
masks=[resize(mask, (418,418), anti_aliasing=False,order=0) for mask in masks_org]
masks_int=[mask.astype(np.uint8) for mask in masks]
mask_stack = np.stack(masks_int, axis=0)

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