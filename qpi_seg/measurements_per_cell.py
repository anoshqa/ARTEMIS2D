#tasks
#import cellpose masks
#import unet masks

import os
import tifffile
import skimage
from skimage.measure import label, regionprops
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import qpi_seg.visualize as vs

#actually you put unseen image
images_folder=r'/mnt/efs/dl_jrc/student_data/S-DC/MIP_unseen_padded'


cellpose_mask_folder=r'/mnt/efs/dl_jrc/student_data/S-DC/Masks_unseen_cellpose'

unet_mask_folder=r'/mnt/efs/dl_jrc/student_data/S-DC/Masks_padded_final'

images=[tifffile.imread(os.path.join(images_folder,file)) for file in os.listdir(images_folder)]
cp_masks=[tifffile.imread(os.path.join(cellpose_mask_folder,file)) for file in os.listdir(cellpose_mask_folder)]
unet_masks=[tifffile.imread(os.path.join(unet_mask_folder,file)) for file in os.listdir(unet_mask_folder)]

#image=tifffile.imread(os.path.join(image_original,'20240316.120455.453.MDA-MB231P-020_HT2D_0.tif_submask_204.png'))
#cp_mask=tifffile.imread(os.path.join(images_folder,'20240316.120455.453.MDA-MB231P-020_HT2D_0.tif_submask_204_cp_masks.tiff'))
#unet_mask=tifffile.imread(os.path.join(unet_mask_folder,'20240316.120455.453.MDA-MB231P-020_HT2D_0.tif_submask_204final.png'))
#vs.visualize(image,unet_mask,unet_mask)


#images to natural RI range
images_RI= [image/1e4 for image in images ]

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
for i in range(1):
    image=images_RI[i]
    cp_mask = cp_masks[i]
    unique_values=np.unique(cp_mask)
    unique_values=unique_values[unique_values>0] #remove background (0)
    mask2=cp_mask.copy()
    for submask_value in unique_values:
        mask2[mask2 != submask_value] = 0
        mask2[mask2>0]=1
        #per cell semantic mask = cp_mask after filter x unet_masks[i]
        combined_mask = mask2 * unet_masks[i]
        combined_mask=combined_mask.astype(np.int64)
        props_table = skimage.measure.regionprops_table(combined_mask, intensity_image=image, properties=['label', 'intensity_mean','area','intensity_std'])
        areas_all.append(np.array(props_table['area']))
        intensity_mean_all.append(np.array(props_table['intensity_mean']))
        intensity_std_all.append(np.array(props_table['intensity_std']))
        
areas=pd.DataFrame(areas_all, columns =['Cell_area','Nucleus_area','Nucleolus_area','Lipid_area'])
meanRI=pd.DataFrame(intensity_mean_all,columns=['Cell_RImean','Nucleus_RImean','Nucleolus_RImean','Lipid_RImean'])
stdRI=pd.DataFrame(intensity_std_all,columns=['Cell_RIdtd','Nucleus_RIstd','Nucleolus_RIstd','Lipid_RIstd'])  
featuresdf=pd.concat([areas,meanRI,stdRI],axis=1)            
#print(featuresdf.head())              

#plt.imshow(cp_mask)
#plt.colorbar()