import matplotlib.pyplot as plt
def visualize(im1,im3):
    plt.figure(figsize=(10, 10))
    plt.subplot(131)
    plt.imshow(im1) 
    plt.title("Input image 1",color="white")
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(im3)
    plt.title("UNet predicted mask",color="white")
    plt.tight_layout()
    plt.axis('off')
    plt.savefig('demoimage.png',dpi=300,transparent=True)