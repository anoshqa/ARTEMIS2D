import napari
import os
import tifffile
import numpy as np
from skimage.transform import resize

# original images
image_folder = r"C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\phenotyping phase\Cropped_MIP"
# all mask files
output_mask_folder = r"C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\phenotyping phase\Cropped_Mask"
# corrected mask folder below
corrected_mask_folder = r"C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\phenotyping phase\Mask_proofread"
# corrected/kept image folder (new — mirrors the surviving images)
corrected_image_folder = r"C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\phenotyping phase\MIP_proofread"

val_image_files = sorted(os.listdir(image_folder))

val_images_org = [tifffile.imread(os.path.join(image_folder, file)) for file in val_image_files]
val_images = [resize(image, (418, 418), order=0, anti_aliasing=False, preserve_range=True) for image in val_images_org]
val_image_stack = np.stack(val_images, axis=0)

mask_files = sorted(os.listdir(output_mask_folder))

masks_org = [tifffile.imread(os.path.join(output_mask_folder, file)).astype(np.uint16) for file in mask_files]
masks = [resize(mask, (418, 418), order=0, anti_aliasing=False, preserve_range=True) for mask in masks_org]
mask_stack = np.stack(masks, axis=0)

print(f"val_image_stack shape: {val_image_stack.shape}")
print(f"mask_stack shape: {mask_stack.shape}")
out_file_name_masks_selected = [os.path.join(corrected_mask_folder, file) for file in mask_files]
out_file_name_images_selected = [os.path.join(corrected_image_folder, file) for file in val_image_files]

# create the viewer and add the image/labels
viewer = napari.Viewer()
viewer.add_image(val_image_stack, name='coins')
labels_layer = viewer.add_labels(mask_stack, name='segmentation')

# slice indices marked for full deletion (image + mask pair skipped on save)
deleted_slices = set()
def current_slice_index():
    return int(viewer.dims.current_step[0])


@viewer.bind_key('d')
def toggle_delete_pair(viewer):
    """Press 'd' to mark/unmark the current image-mask PAIR for deletion."""
    idx = current_slice_index()
    if idx in deleted_slices:
        deleted_slices.discard(idx)
        print(f"Un-marked pair {idx} ({val_image_files[idx]} / {mask_files[idx]})")
    else:
        deleted_slices.add(idx)
        print(f"Marked pair {idx} for deletion: {val_image_files[idx]} / {mask_files[idx]}")



napari.run()
saved, skipped = 0, 0
for i in range(len(out_file_name_masks_selected)):
    if i in deleted_slices:
        print(f"Skipping deleted pair {i}: {val_image_files[i]} / {mask_files[i]}")
        skipped += 1
        continue
    image=masks_org[i]
    edited_mask=mask_stack[i]
    w=image.shape[0]
    h=image.shape[1]
    mask_resized=resize(edited_mask, (w,h),order=0,anti_aliasing=False)
    tifffile.imwrite(
        out_file_name_masks_selected[i],
        mask_resized
    )

    # copy over the matching original image, unresized
    tifffile.imwrite(out_file_name_images_selected[i], val_images_org[i])

    saved += 1

print(f"Saved {saved} pairs, skipped {skipped} deleted pair(s): "
      f"{[(val_image_files[i], mask_files[i]) for i in sorted(deleted_slices)]}")