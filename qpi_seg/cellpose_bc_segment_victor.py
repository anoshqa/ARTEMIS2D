##CELLPOSE TESTING SCRIPT- export cell instance masks on completely unseen images :) 

from cellpose import io
from cellpose import models, core, io, plot, train, metrics
import torch
import numpy as np
import os
import tifffile
from skimage.transform import resize
import napari

cuda_available = torch.cuda.is_available()
print(f"Is CUDA available? {cuda_available}")
first_model_path=r"D:\Cellpose_segmentation\cpmodel_test_all_images_50epochs.pt"

#put image folder name
image_folder = r"F:\TRAINING_DATA_FINAL\Remaining_MIP_from_storage"
#put output mask folder name
output_mask_folder = r"F:\TRAINING_DATA_FINAL\Remaining_MIP_storage_cp_masks"
#all val image files
val_image_files=os.listdir(image_folder)

#reads tiff files
val_images=[tifffile.imread(os.path.join(image_folder, file)) for file in val_image_files]

out_file_name_stems=[os.path.splitext(file)[0][:61]+'_cp_masks.tiff'for file in val_image_files]

out_file_name_masks=[os.path.join(output_mask_folder, file) for file in out_file_name_stems]
#keep GPU=true
cpmodel_baseline_50epochs = models.CellposeModel(gpu=True,
                                pretrained_model=first_model_path)
test_masks_resized=[]
resized_val_images=[]
cropped_masks=[]
w_list=[]
h_list=[]
out_file_name_masks_selected=[]
for i in range(300,552):
   
    w=val_images[i].shape[0]
    h=val_images[i].shape[1]
    w_list.append(w)
    h_list.append(h)
    #print(w,h)
    image = val_images[i]
    #smaller sizes give faster output - training was done with (418,418)
    val_image_resized=resize(image, (418,418), anti_aliasing=True,preserve_range=True) 
    resized_val_images.append(val_image_resized)
    #niter needs to be higher if your cells are bigger
    test_masks_output, flows, styles = cpmodel_baseline_50epochs.eval(val_image_resized, batch_size=4, normalize = True,flow_threshold=0.4)
    cropped_masks.append(test_masks_output)
    out_file_name_masks_selected.append(out_file_name_masks[i])
    #check resize function's preserve_range, interpolation order options
    #test_masks_resized.append(resize(test_masks_output, (w,h),order=0,               # 0 corresponds to nearest-neighbor
    #anti_aliasing=False))
    #tifffile.imwrite(
    #    out_file_name_masks[i],
    #    test_masks_resized[i]
    #)
val_image_stack = np.stack(resized_val_images, axis=0)
mask_stack = np.stack(cropped_masks, axis=0)
if __name__ == '__main__':


    viewer = napari.Viewer()
# create the viewer and add the coins image
    viewer.add_image(val_image_stack, name='mip')
# add the labels
    viewer.add_labels(mask_stack, name='segmentation')
    napari.run()
for i in range(len(out_file_name_masks_selected)):
   
    w=w_list[i]
    h=h_list[i]
    edited_mask=mask_stack[i]
    test_masks_resized.append(resize(edited_mask, (w,h),order=0,anti_aliasing=False))
    tifffile.imwrite(
        out_file_name_masks_selected[i],
        test_masks_resized[i]
    )

#visualize.visualize(val_images[0], test_masks_resized[0])
#if 5 masks are not there use
