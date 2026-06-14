import matplotlib.pyplot as plt
def visualize(im1,im2):
    plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.imshow(im1)  # TODO
    plt.subplot(122)
    plt.imshow(im2)  # TODO
    plt.tight_layout()