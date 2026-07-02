import os
import qpi_seg.file_charactersmatch as filetest
image_folder=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\MIP_stitched_with_multiple_cells'
mask_folder=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\Reference_masks_for_mask_stitched_with_multiple_cells'
image_filenames=sorted(os.listdir(image_folder))
mask_filenames=sorted(os.listdir(mask_folder))
print(filetest.filetest(image_filenames,mask_filenames))
