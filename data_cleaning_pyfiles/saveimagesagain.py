#this script has only one purpose - to repeat the image for which there are many cell masks so that we can compare 
import os
import tifffile
import skimage
from skimage.measure import label, regionprops
import numpy as np
import pandas as pd
image_folder=r"C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\UNSEEN_MIP_1"
cellpose_mask_folder=r"C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\UNSEEN_MIP_1_CELL_MASK"

#also stores (418,418) for easier visualization
output_folder=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\UNSEEN_MIP_1_REPEATED'
output_folder_org_size=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\UNSEEN_MIP_1_ORIGINAL_SIZE'
images=[skimage.io.imread(os.path.join(image_folder,file)) for file in sorted(os.listdir(image_folder))]
resized_images=[skimage.transform.resize(image, (418,418), anti_aliasing=True,preserve_range=True) for image in images]
cp_masks=[skimage.io.imread(os.path.join(cellpose_mask_folder,file)) for file in sorted(os.listdir(cellpose_mask_folder))]
out_file_name_stems=os.listdir(image_folder)
out_file_names=[]

print(f"len(images): {len(images)}")
print(f"len(resized_images): {len(resized_images)}")
print(f"len(cp_masks): {len(cp_masks)}")
resized_list=[]
org_list=[]
for i in range(len(cp_masks)):
    image=resized_images[i]
    cp_mask = cp_masks[i]
    unique_values=np.unique(cp_mask)
    unique_values=unique_values[unique_values>0] #remove background (0)
    print(unique_values)
    for submask_value in unique_values:
        out_file_name_stem=f"{out_file_name_stems[i]}_mip{submask_value}.tiff"
        out_file_names.append(out_file_name_stem)
        resized_list.append(resized_images[i])
        org_list.append(images[i])
for i in range(len(out_file_names)):
    tifffile.imwrite(
        os.path.join(output_folder,out_file_names[i]),
        resized_list[i]
    )
    tifffile.imwrite(
        os.path.join(output_folder_org_size,out_file_names[i]),
        org_list[i]
    )