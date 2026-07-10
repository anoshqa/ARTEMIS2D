import os
import data_cleaning_pyfiles.file_charactersmatch as filetest
image_folder=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\phenotyping phase\MIP'
mask_folder=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\phenotyping phase\Combined_mask'
image_filenames=sorted(os.listdir(image_folder))
mask_filenames=sorted(os.listdir(mask_folder))
print(filetest.filetest(image_filenames,mask_filenames))
