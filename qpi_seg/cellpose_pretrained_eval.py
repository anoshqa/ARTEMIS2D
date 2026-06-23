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

first_model_path=r"/mnt/efs/dl_jrc/student_data/S-DC/model/models/cpmodel_test_all_images_50epochs"

#image_folder = r'/mnt/efs/dl_jrc/student_data/S-DC/MIP_unseen_padded'
image_folder = r'/mnt/efs/dl_jrc/student_data/S-DC/Test_Swati/Save_mask_test'

output_mask_folder = r'/mnt/efs/dl_jrc/student_data/S-DC/Unet_output_swati'

model = models.CellposeModel(gpu=True)
val_image_files=os.listdir(image_folder)


val_images=[tifffile.imread(os.path.join(image_folder, file)) for file in val_image_files]
#save width and height of val_images[0]
w=val_images[0].shape[0]
h=val_images[0].shape[1]

out_file_name_stems=[os.path.splitext(file)[0][:30]+'_cp_masks.tiff'for file in val_image_files]


val_image_resized=[resize(image, (418,418), anti_aliasing=True,preserve_range=True) for image in val_images]

cpmodel_baseline_50epochs = models.CellposeModel(gpu=True,
                                pretrained_model=first_model_path)

test_masks_output, flows, styles = cpmodel_baseline_50epochs.eval(val_image_resized, batch_size=4, normalize = True,niter=3000,flow_threshold=1)

#TODO: I have to change shape
#add a function to resize to the original shape of the image


test_masks_resized=[resize(image, (w,h),preserve_range=True,order=0) for image in test_masks_output]

out_file_name_masks=[os.path.join(output_mask_folder, file) for file in out_file_name_stems]

for i in range(len(out_file_name_masks)):
   tifffile.imwrite(
        out_file_name_masks[i],
        test_masks_resized[i]
    )

#for visualizing first five masks
#import qpi_seg.plot_grids as pg
#val_image_resized_5=val_image_resized[0:5]
#test_masks=test_masks_output[0:5]
#pg.plot_grids(val_image_resized,test_masks)

visualize.visualize(val_images[0], test_masks_resized[0])
#if 5 masks are not there use
