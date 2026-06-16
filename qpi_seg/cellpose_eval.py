import numpy as np
from cellpose import models, core, io, plot
from pathlib import Path
from tqdm import trange
import matplotlib.pyplot as plt
from natsort import natsorted
import os
import qpi_seg.dataset
#import qpi_seg.plot_grids as pg
save_mask_dir = '/mnt/efs/dl_jrc/student_data/S-DC/Cellpose_masks'
image_folder=r'/mnt/efs/dl_jrc/student_data/S-DC/MIP_padded_final'
image_filenames=sorted(os.listdir(image_folder))

image_files=[os.path.join(image_folder,image_filenames[i]) for i in range(len(image_filenames))]
model = models.CellposeModel(gpu=True)
flow_threshold = 0
cellprob_threshold = 0.3
tile_norm_blocksize = 0
#img_normalization
norm_min=13300
norm_max=14100
#masks, flows, styles = model.eval(img_norm, diameter= 400, batch_size=32,channels=[[0,0]],flow_threshold=0, cellprob_threshold=0,normalize={"tile_norm_blocksize": tile_norm_blocksize})
masks_ext = ".png"
mask_list=[]
image_list=[]
flow_list=[]
for i in trange(5):
    f = image_files[i]
    img = io.imread(f)
    img_norm= (img - norm_min) / (norm_max - norm_min)
    masks, flows, styles = model.eval(img_norm, n_iter=2000,diameter=400, batch_size=32, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
                                  normalize=False)
    io.imsave(os.path.join(save_mask_dir,os.path.splitext(image_filenames[i])[0] + "_masks" + masks_ext), masks)
    mask_list.append(masks)
    image_list.append(img_norm)
    flow_list.append(flows)
imgs=[io.imread(f) for f in image_files]

#pg.plot_grids(imgs, masks)
fig = plt.figure(figsize=(12,5))
plot.show_segmentation(fig, img_norm, masks, flows[0])
#plt.savefig('segmentation_test.png')
#plt.tight_layout()