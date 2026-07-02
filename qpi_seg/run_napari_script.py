import napari
import os
import tifffile
import numpy as np

image_folder = r"H:\Victor data\QPI\TIFF 20260611 Exp2 Res t2\2D"
#put output mask folder name
output_mask_folder = r"H:\Victor data\QPI\TIFF 20260611 Exp2 Res t2\2D_mask_cp_1"
#all val image files


val_image_files=os.listdir(image_folder)
val_images=[tifffile.imread(os.path.join(image_folder, file)) for file in val_image_files[0:5]]
val_image_stack = np.stack(val_images, axis=0)
mask_files=os.listdir(output_mask_folder)
masks=[tifffile.imread(os.path.join(output_mask_folder, file)) for file in mask_files[0:5]]
mask_stack = np.stack(masks, axis=0)
if __name__ == '__main__':


    viewer = napari.Viewer()
# create the viewer and add the coins image
    viewer.add_image(val_image_stack, name='coins')
# add the labels
    viewer.add_labels(mask_stack, name='segmentation')
    napari.run()