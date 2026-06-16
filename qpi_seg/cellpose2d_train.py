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
def graph_losses(train_losses, test_losses):
    plt.figure(figsize=(6, 4), dpi=150)

    plt.plot(train_losses, label="training loss")
    plt.plot(test_losses, label="test loss")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training vs test loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"training_test_loss-{timestamp}.png")
if __name__ == "__main__":
    # USER INPUT VALUES =====================================
    n_epochs = 50
    learning_rate = 1e-5
    weight_decay = 0.1
    batch_size = 1
    model_name = "cpmodel_baseline_50epochs"
    # Data Paths
    image_folder = r'/mnt/efs/dl_jrc/student_data/S-DC/MIP_padded_final'
    mask_folder = r'/mnt/efs/dl_jrc/student_data/S-DC/Binary_masks/Binary_masks_'
    save_path = Path("/mnt/efs/dl_jrc/student_data/S-DC/model")
    train_files_percentage = 0.8
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
    io.logger_setup()

    first_model_path, train_losses, test_losses = train.train_seg(model.net,
                                                                train_data=train_image_resized,
                                                                train_labels=train_masks_resized,
                                                                test_data=val_images_resized,
                                                                test_labels=val_masks_resized,
                                                                batch_size=batch_size,
                                                                n_epochs=n_epochs,
                                                                learning_rate=learning_rate,
                                                                weight_decay=weight_decay,
                                                                nimg_per_epoch=max(2, len(train_images)), # can change this
                                                                model_name=model_name, 
                                                                normalize = True, 
                                                                save_path = save_path, save_every = 5,
                                                                save_each = True
                                                                )
    cpmodel_baseline_50epochs = models.CellposeModel(gpu=True,
                                pretrained_model=first_model_path)
    test_masks_output, flows, styles = cpmodel_baseline_50epochs.eval(val_images, batch_size=32, normalize = False)# batch size here is the maximunumber of masks 
    ap, tp, fp, fn = metrics.average_precision(val_masks, test_masks_output)
    print(f'average precision at iou threshold 0.5 = {ap}, \n false positives = {fp}, \n false negatives = {fn}, \n true positives = {tp}')
    print(f'>>> average precision at iou threshold 0.5 = {ap[:,0].mean():.3f}')

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    np.savez(save_path/f"test_results_{timestamp}.npz", ap=ap, tp=tp, fp=fp, fn=fn, test_masks=val_masks, test_masks_output=test_masks_output)
    
    # %%
    for i in range(len(val_images)):
        visual_test.visualize(val_images[i], val_masks[i], test_masks_output[i])
    
    graph_losses(train_losses, test_losses)