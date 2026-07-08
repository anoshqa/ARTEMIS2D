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


cellpose_mask_folder=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\UNSEEN_MIP_1_CELL_MASK'

unet_mask_folder=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\UNSEEN_UNET_MASK'

output_folder=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\combined_mask_2'
#images=[skimage.io.imread(os.path.join(images_folder,file)) for file in sorted(os.listdir(images_folder))]
cp_masks=[skimage.io.imread(os.path.join(cellpose_mask_folder,file)) for file in sorted(os.listdir(cellpose_mask_folder))]
unet_masks=[skimage.io.imread(os.path.join(unet_mask_folder,file)) for file in sorted(os.listdir(unet_mask_folder))]

out_file_name_stems=[os.path.splitext(file)[0][:40]+'_combined_masks'for file in sorted(os.listdir(unet_mask_folder))]

#images to natural RI range
#images_RI= [image/1e4 for image in images ]

intensity_means_all = []
areas_all=[]

#pixel to um conversion
# 0.095 x 0.095 um
def intensity_std(mask, intensity):
    return np.std(intensity[mask])

combined_masks=[]
intensity_mean_all=[]
area_all=[]
intensity_std_all=[]

out_file_name_masks=[]
for i in range(len(cp_masks)):
    #image=images_RI[i]
    cp_mask = cp_masks[i]
    unique_values=np.unique(cp_mask)
    unique_values=unique_values[unique_values>0] #remove background (0)
    print(unique_values)
    for submask_value in unique_values:
        mask2=cp_masks[i].copy()
        unet_mask=unet_masks[i].copy()
        unet_mask[mask2 != submask_value] = 0
        mask2[mask2 !=submask_value]=0
        mask2[mask2 >0]=1
        unet_mask[(unet_mask==0) & (mask2==1)]=1
        #per cell semantic mask = cp_mask after filter x unet_masks[i]
        out_file_name_stem=f"{out_file_name_stems[i]}_mask{submask_value}.tiff"
        print(out_file_name_stem)
        out_file_name_masks.append(os.path.join(output_folder,out_file_name_stem))
        combined_masks.append(unet_mask)
        #props_table = skimage.measure.regionprops_table(combined_mask, intensity_image=image, properties=['label', 'intensity_mean','area','intensity_std'])
        #areas_all.append(np.array(props_table['area']))
        #intensity_mean_all.append(np.array(props_table['intensity_mean']))
        #intensity_std_all.append(np.array(props_table['intensity_std']))
for i in range(len(out_file_name_masks)):
    tifffile.imwrite(
        out_file_name_masks[i],
        combined_masks[i].astype(np.uint16)
    )
#areas=pd.DataFrame(areas_all, columns =['Cell_area','Nucleus_area','Nucleolus_area','Lipid_area'])
#meanRI=pd.DataFrame(intensity_mean_all,columns=['Cell_RImean','Nucleus_RImean','Nucleolus_RImean','Lipid_RImean'])
#stdRI=pd.DataFrame(intensity_std_all,columns=['Cell_RIdtd','Nucleus_RIstd','Nucleolus_RIstd','Lipid_RIstd'])  
#featuresdf=pd.concat([areas,meanRI,stdRI],axis=1)            
#print(featuresdf.head())              

#plt.imshow(cp_mask)
#plt.colorbar()