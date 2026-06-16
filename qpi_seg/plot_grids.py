import matplotlib.pyplot as plt
import numpy as np
import qpi_seg.mask_overlay as mask_overlay
from cellpose import plot
fig,ax = plt.subplots(5,13,figsize=(13,5.7))

def plot_grids(imgs, masks):
    for i in range(5):
        for j in range(13):
            idx=i*13+j
            maski = masks[idx]
            img0 = imgs[idx].copy()

            if img0.shape[0] < 4:
                img0 = np.transpose(img0, (1, 2, 0))
            if img0.shape[-1] < 3 or img0.ndim < 3:
                img0 = plot.image_to_rgb(img0, channels=[0,0])
            else:
                if img0.max() <= 50.0:
                    img0 = np.uint8(np.clip(img0, 0, 1) * 255)
        overlay=mask_overlay(img0,maski)
        ax[i][j].imshow(overlay)
plt.setp(plt.gcf().get_axes(),xticks=[],yticks=[])
plt.savefig('grids_test.png')
#plt.show()