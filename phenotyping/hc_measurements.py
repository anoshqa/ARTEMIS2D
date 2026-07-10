import os
import tifffile
import skimage
from skimage.measure import label, regionprops
import numpy as np
import pandas as pd
images_folder=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\phenotyping phase\MIP_proofread'


combined_mask_folder=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\phenotyping phase\Mask_proofread'

images=[skimage.io.imread(os.path.join(images_folder,file)) for file in sorted(os.listdir(images_folder))]

masks=[skimage.io.imread(os.path.join(combined_mask_folder, maskfile) for maskfile in sorted(os.listdir(combined_mask_folder)))]

#images to natural RI range
images_RI= [image/1e4 for image in images ]

intensity_means_all = []
areas_all=[]

#pixel to um conversion
# 0.095 x 0.095 um
def intensity_std(mask, intensity):
    return np.std(intensity[mask])

intensity_mean_all=[]
intensity_std_all=[]
for i in range(len(masks)):
    image=images_RI[i]
    combined_mask=masks[i]
    props_table = skimage.measure.regionprops_table(combined_mask, intensity_image=image, properties=['label', 'intensity_mean','area','intensity_std'])
    areas_all.append(np.array(props_table['area']))
    intensity_mean_all.append(np.array(props_table['intensity_mean']))
    intensity_std_all.append(np.array(props_table['intensity_std']))
areas=pd.DataFrame(areas_all, columns =['Cell_area','Nucleus_area','Nucleolus_area','Lipid_area'])
meanRI=pd.DataFrame(intensity_mean_all,columns=['Cell_RImean','Nucleus_RImean','Nucleolus_RImean','Lipid_RImean'])
stdRI=pd.DataFrame(intensity_std_all,columns=['Cell_RIdtd','Nucleus_RIstd','Nucleolus_RIstd','Lipid_RIstd'])  
featuresdf=pd.concat([areas,meanRI,stdRI],axis=1)            
print(featuresdf.head())              
featuresdf.to_csv('hc_features.csv')