#tasks
#imports cellpose masks
#imports unet masks

import os
import tifffile
import skimage
from skimage.measure import label, regionprops
import numpy as np
import pandas as pd
#actually you put unseen image
#images_folder=r'D:\TRAINING_DATA_FINAL\TEST_MIP'


cellpose_mask_folder=r'D:\TRAINING_DATA_FINAL\Remaining_MIP_storage_cp_masks_corrected'

unet_mask_folder=r'D:\TRAINING_DATA_FINAL\Remaining_MIP_storage_unet_masks'

output_folder=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\Remaining_MIP_combined_mask1'
#images=[skimage.io.imread(os.path.join(images_folder,file)) for file in sorted(os.listdir(images_folder))]
cp_files=sorted(os.listdir(cellpose_mask_folder))
unet_files=sorted(os.listdir(unet_mask_folder))

out_file_name_stems=[os.path.splitext(file)[0][:40]+'_combined_masks'for file in unet_files]

#images to natural RI range
#images_RI= [image/1e4 for image in images ]

intensity_means_all = []
areas_all=[]

#pixel to um conversion
# 0.095 x 0.095 um
def intensity_std(mask, intensity):
    return np.std(intensity[mask])


for i in range(len(cp_files)):
    #image=images_RI[i]
    cp_mask = skimage.io.imread(os.path.join(cellpose_mask_folder,cp_files[i]))
    unet_mask=skimage.io.imread(os.path.join(unet_mask_folder,unet_files[i]))
    unique_values=np.unique(cp_mask)
    unique_values=unique_values[unique_values>0] #remove background (0)
    print(unique_values)
    for submask_value in unique_values:
        mask2=cp_mask.copy()
        unet_mask2=unet_mask.copy()
        unet_mask2[mask2 != submask_value] = 0
        mask2[mask2 !=submask_value]=0
        mask2[mask2 >0]=1
        unet_mask2[(unet_mask2==0) & (mask2==1)]=1
        #per cell semantic mask = cp_mask after filter x unet_masks[i]
        out_file_name_stem=f"{out_file_name_stems[i]}_mask{submask_value}.tiff"
        tifffile.imwrite(
            os.path.join(output_folder,out_file_name_stem),
            unet_mask2.astype(np.uint16)
        )#props_table = skimage.measure.regionprops_table(combined_mask, intensity_image=image, properties=['label', 'intensity_mean','area','intensity_std'])
        #areas_all.append(np.array(props_table['area']))
        #intensity_mean_all.append(np.array(props_table['intensity_mean']))
        #intensity_std_all.append(np.array(props_table['intensity_std']))
#areas=pd.DataFrame(areas_all, columns =['Cell_area','Nucleus_area','Nucleolus_area','Lipid_area'])
#meanRI=pd.DataFrame(intensity_mean_all,columns=['Cell_RImean','Nucleus_RImean','Nucleolus_RImean','Lipid_RImean'])
#stdRI=pd.DataFrame(intensity_std_all,columns=['Cell_RIdtd','Nucleus_RIstd','Nucleolus_RIstd','Lipid_RIstd'])  
#featuresdf=pd.concat([areas,meanRI,stdRI],axis=1)            
#print(featuresdf.head())              

#plt.imshow(cp_mask)
#plt.colorbar()