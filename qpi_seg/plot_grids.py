import matplotlib.pyplot as plt
import numpy as np
fig,ax = plt.subplots(2,5,figsize=(10,5))
def plot_grids(imgs,masks):
    fig,ax = plt.subplots(2,5,figsize=(10,5))
    for j in range(5):
        ax[0,j].imshow(imgs[j])
        ax[1,j].imshow(masks[j])
            
    plt.setp(plt.gcf().get_axes(),xticks=[],yticks=[])
    plt.tight_layout()
    plt.savefig('grids_test.png')