from cellpose import train
from cellpose import io
from cellpose import models, core, io, plot, train, metrics
import torch
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path
import qpi_seg.visualize as visual_test
import os
import tifffile
from datetime import datetime
import torchvision.transforms.v2 as transforms_v2
import qpi_seg.visualize as visual_test
from skimage.transform import resize
import qpi_seg.plot_grids as pg

image_folder = r'/mnt/efs/dl_jrc/student_data/S-DC/MIP_padded_final'
mask_folder = r'/mnt/efs/dl_jrc/student_data/S-DC/Binary_masks/Binary_masks_'
train_files_percentage = 0.95
image_filenames=sorted(os.listdir(image_folder))
mask_filenames=sorted(os.listdir(mask_folder))
npimage_files=np.array(image_filenames)
npmask_files=np.array(mask_filenames)
model = models.CellposeModel(gpu=True)
shuffled_indices = np.random.permutation(len(npimage_files))
image_files = npimage_files[shuffled_indices] # YOUR CODE HERE -> Hint: first transform to np.array
mask_files = npmask_files[shuffled_indices]
tot_num_image_files = len(npimage_files)
tot_num_mask_files = len(npmask_files)
num_train_image_files = int(train_files_percentage*tot_num_image_files) 
num_train_mask_files = int(train_files_percentage*tot_num_mask_files)
train_image_files = image_files[:num_train_image_files ] # YOUR CODE HERE
train_mask_files = mask_files[:num_train_mask_files ] # YOUR CODE HERE
val_image_files = image_files[num_train_image_files:] # YOUR CODE HERE
val_mask_files = mask_files[num_train_mask_files:] # YOUR CODE HERE
transform= transforms_v2.Resize((128,128),interpolation=transforms_v2.InterpolationMode.NEAREST)
    
train_images = [tifffile.imread(os.path.join(image_folder, file)) for file in train_image_files]
train_masks=[tifffile.imread(os.path.join(mask_folder, file)) for file in train_mask_files]
val_images=[tifffile.imread(os.path.join(image_folder, file)) for file in val_image_files]
val_masks=[tifffile.imread(os.path.join(mask_folder, file))for file in val_mask_files]
    
train_image_resized=[resize(image, (418,418), anti_aliasing=True) for image in train_images]
train_masks_resized=[resize(image, (418,418), anti_aliasing=True) for image in train_masks]
val_image_resized=[resize(image, (418,418), anti_aliasing=True) for image in val_images]
val_masks_resized=[resize(image, (418,418), anti_aliasing=True) for image in val_masks]

first_model_path=r"/mnt/efs/dl_jrc/student_data/S-DC/model/models/cpmodel_baseline_50epochs"


cpmodel_baseline_50epochs = models.CellposeModel(gpu=True,
                                pretrained_model=first_model_path)
test_masks_output, flows, styles = cpmodel_baseline_50epochs.eval(val_image_resized, batch_size=4, normalize = True)
#ap, tp, fp, fn = metrics.average_precision(val_masks_resized, test_masks_output)
#print(f'average precision at iou threshold 0.5 = {ap}, \n false positives = {fp}, \n false negatives = {fn}, \n true positives = {tp}')
#print(f'>>> average precision at iou threshold 0.5 = {ap[:,0].mean():.3f}')
import qpi_seg.plot_grids as pg
val_image_resized_5=val_image_resized[5:10]
test_masks=test_masks_output[5:10]
pg.plot_grids(val_image_resized,test_masks)