##CELLPOSE TESTING SCRIPT- export cell instance masks on completely unseen images :) 

from cellpose import io
from cellpose import models, core, io, plot, train, metrics
import torch
import numpy as np
import matplotlib.pyplot as plt
import qpi_seg.visualize as visual_test
import os
import tifffile
import torchvision.transforms.v2 as transforms_v2
from skimage.transform import resize
import qpi_seg.visualize_unseen_unmasked as visualize

cuda_available = torch.cuda.is_available()
print(f"Is CUDA available? {cuda_available}")
first_model_path=r"D:\Cellpose_segmentation\cpmodel_test_all_images_50epochs.pt"

#put image folder name
image_folder = r"H:\Victor data\QPI\TIFF 20260611 Exp2 Res t2\2D"
#put output mask folder name
output_mask_folder = r"H:\Victor data\QPI\TIFF 20260611 Exp2 Res t2\2D_mask_cp_1"
#all val image files
val_image_files=os.listdir(image_folder)

#reads tiff files
val_images=[tifffile.imread(os.path.join(image_folder, file)) for file in val_image_files]

out_file_name_stems=[os.path.splitext(file)[0][:61]+'_cp_masks.tiff'for file in val_image_files]

out_file_name_masks=[os.path.join(output_mask_folder, file) for file in out_file_name_stems]
#keep GPU=true
cpmodel_baseline_50epochs = models.CellposeModel(gpu=True,
                                pretrained_model=first_model_path)
test_masks_resized=[]
for i in range(len(val_images)):
   
    w=val_images[i].shape[0]
    h=val_images[i].shape[1]
    print(w,h)
    image = val_images[i]
    #smaller sizes give faster output - training was done with (418,418)
    val_image_resized=resize(image, (418,418), anti_aliasing=True,preserve_range=True) 
    #niter needs to be higher if your cells are bigger
    test_masks_output, flows, styles = cpmodel_baseline_50epochs.eval(val_image_resized, batch_size=4, normalize = True,flow_threshold=0)

    #check resize function's preserve_range, interpolation order options
    test_masks_resized.append(resize(test_masks_output, (w,h),order=0,               # 0 corresponds to nearest-neighbor
    anti_aliasing=False))
    tifffile.imwrite(
        out_file_name_masks[i],
        test_masks_resized[i]
    )


#for visualizing first five masks
#import qpi_seg.plot_grids as pg
#val_image_resized_5=val_image_resized[0:5]
#test_masks=test_masks_output[0:5]
#pg.plot_grids(val_image_resized,test_masks)

#visualize.visualize(val_images[0], test_masks_resized[0])
#if 5 masks are not there use
