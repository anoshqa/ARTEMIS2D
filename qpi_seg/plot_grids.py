import matplotlib.pyplot as plt
import numpy as np
import qpi_seg.mask_overlay as mask_overlay
from cellpose import plot
fig,ax = plt.subplots(5,5,figsize=(10,5))

def plot_grids(imgs, masks):
    for i in range(5):
        for j in range(5):
            idx=i*5+j
            maski = masks[idx]
            img0 = imgs[idx].copy()
            
            
plt.setp(plt.gcf().get_axes(),xticks=[],yticks=[])
plt.savefig('grids_test.png')
#plt.show()