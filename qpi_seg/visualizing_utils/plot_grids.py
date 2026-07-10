import matplotlib.pyplot as plt
import numpy as np
fig,ax = plt.subplots(2,5,figsize=(10,5))
def plot_grids(imgs,masks):
    fig,ax = plt.subplots(2,5,figsize=(10,5))
    for j in range(5):
        ax[0,j].imshow(imgs[j])
        ax[1,j].imshow(masks[j])
    ax[0,0].set_title("Ch1 (Background)",color='white')
    ax[0,1].set_title("Ch2 (Cytoplasm)",color='white')
    ax[0,2].set_title("Ch3 (Nucleoplasm)",color='white')
    ax[0,3].set_title("Ch4 (Nucleoli)",color='white')
    ax[0,4].set_title("Ch5 (Lipid Droplets)",color='white')       
    plt.setp(plt.gcf().get_axes(),xticks=[],yticks=[])
    plt.tight_layout()
    plt.savefig('grids_test.png',transparent=True)