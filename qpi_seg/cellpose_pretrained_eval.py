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


first_model_path=r"/mnt/efs/dl_jrc/student_data/S-DC/model/models/cpmodel_baseline_50epochs"

image_folder = r'/mnt/efs/dl_jrc/student_data/S-DC/MIP_unseen_padded'

output_mask_folder = r'/mnt/efs/dl_jrc/student_data/S-DC/Masks_unseen_cellpose'

model = models.CellposeModel(gpu=True)
val_image_files=os.listdir(image_folder)

#val_image_files=['','','','','']

val_images=[tifffile.imread(os.path.join(image_folder, file)) for file in val_image_files]

out_file_name_stems=[os.path.splitext(file)[0]+'_cp_masks.tiff'for file in val_image_files ]


val_image_resized=[resize(image, (418,418), anti_aliasing=True) for image in val_images]

cpmodel_baseline_50epochs = models.CellposeModel(gpu=True,
                                pretrained_model=first_model_path)

test_masks_output, flows, styles = cpmodel_baseline_50epochs.eval(val_image_resized, batch_size=4, normalize = True,niter=2000)


test_masks_resized=[resize(image, (836,836), anti_aliasing=True) for image in test_masks_output]

out_file_name_masks=[os.path.join(output_mask_folder, file) for file in out_file_name_stems]

for i in range(len(out_file_name_masks)):
    tifffile.imwrite(
        out_file_name_masks[i],
        test_masks_resized[i]
    )

#for visualiing first five masks
import qpi_seg.plot_grids as pg
val_image_resized_5=val_image_resized[0:5]
test_masks=test_masks_output[0:5]
pg.plot_grids(val_image_resized,test_masks)