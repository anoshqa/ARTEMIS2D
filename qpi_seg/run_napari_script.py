import napari
import os
import tifffile
import numpy as np
from skimage.transform import resize

#original  images
image_folder = r"D:\TRAINING_DATA_FINAL\Remaining_MIP_repeated_resized"
#all mask files 
output_mask_folder = r"C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\Remaining_MIP_combined_mask1"
#corrected mask folder below
corrected_mask_folder=r"C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\Remaining_MIP_combined_mask_corrected"

val_image_files=sorted(os.listdir(image_folder))[1000:]

val_images_org=[tifffile.imread(os.path.join(image_folder, file)) for file in val_image_files]
val_images=[resize(image, (418,418),order=0, anti_aliasing=False,preserve_range=True) for image in val_images_org]
val_image_stack = np.stack(val_images, axis=0)

mask_files=sorted(os.listdir(output_mask_folder))[1000:]

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
    image=masks_org[i]
    edited_mask=mask_stack[i]
    w=image.shape[0]
    h=image.shape[1]
    mask_resized=resize(edited_mask, (w,h),order=0,anti_aliasing=False)
    tifffile.imwrite(
        out_file_name_masks_selected[i],
        mask_resized
    )