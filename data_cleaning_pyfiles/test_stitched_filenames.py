import os
import qpi_seg.file_charactersmatch as filetest
image_folder=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\UNSEEN_UNET_MASK'
mask_folder=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\UNSEEN_MIP_1_CELL_MASK'
image_filenames=sorted(os.listdir(image_folder))
mask_filenames=sorted(os.listdir(mask_folder))
print(filetest.filetest(image_filenames,mask_filenames))
