##UNET TESTING SCRIPT- export a merged mask of 5 channels on completely unseen images :) 
import torch
import tifffile
import os
import torchvision.transforms.v2 as transforms_v2
from models.unet import UNet
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import qpi_seg.visualizing_utils.plot_grids as pg
import qpi_seg.train.split_mask_5_channels as split
import qpi_seg.visualizing_utils.visualize_unseen_unmasked as visualize
from skimage.transform import resize

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#assert torch.cuda.is_available()
model=UNet(depth=6,in_channels=1,out_channels=5, num_fmaps=32,final_activation=nn.Softmax()).to(device)

#change the model path

#for personal PC -
model_path=r"C:\Users\anous\OneDrive - Johns Hopkins\2026_DL_Janelia_course\UNet_model_1\checkpoint_epoch_190.pt"
#add map_location if using on CPU


model_save=torch.load(model_path,map_location=torch.device('cpu'))

#if using GPU uncomment the following two lines
#assert torch.cuda.is_available()
#model_save=torch.load(model_path)

model.load_state_dict(model_save['model_state_dict'])
model=model.to(device)

#put your unseen images here
test_images_folder=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\Test_Santosh_MIP'

#put folder where masks will be saved
unet_masks_output_folder=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\Test_Santosh_MIP_Output'


test_files= os.listdir(test_images_folder)
test_images = [tifffile.imread(os.path.join(test_images_folder,file)) for file in test_files]
#print(test_images[0].shape)

out_file_name_stems=[os.path.splitext(file)[0][:30]+'_unet_masks.tiff'for file in test_files ]

out_file_name_masks=[os.path.join(unet_masks_output_folder, file) for file in out_file_name_stems]

from_np=transforms_v2.Lambda(lambda x: torch.from_numpy(x))

#for victor's dataset may have to play with this
norm_min=13370
norm_max=14200
model.eval()
model = model.to(device)
clustermap=[]
clustermap_list=[]
scale= 0.2222 / 0.095 
for i in range(len(test_files)):
    w=test_images[i].shape[0]
    h=test_images[i].shape[1]
    new_h, new_w = round(h*scale), round(w*scale)
    img_resampled = resize(test_images[i], (new_h, new_w), anti_aliasing=True, preserve_range=True)
    # pad up to the smallest multiple of 32 that still contains the full
    # image, per image, so nothing is truncated regardless of input size
    im_size = 32 * ((max(new_w, new_h) + 31) // 32)
    
    transform = transforms_v2.CenterCrop((im_size, im_size))
    cropped_transform=transforms_v2.CenterCrop((new_w,new_h))
    torch_test_image = from_np(img_resampled)
    torch_test_image=torch_test_image.float()
    img_norm= (torch_test_image - norm_min) / (norm_max - norm_min)
    img_norm2=img_norm.unsqueeze(dim=0).to(device)
    img_norm2=transform(img_norm2.unsqueeze(dim=0).to(device))
    torch_prediction=model(img_norm2)
    clustermap=np.zeros((im_size,im_size))
    torch_prediction=torch_prediction.reshape(5,im_size,im_size).detach().cpu()
    ch1_prediction = torch_prediction[0,:,:]>=0.5
    ch2_prediction = torch_prediction[1,:,:]>=0.5
    ch3_prediction= torch_prediction[2,:,:]>=0.5
    ch4_prediction=torch_prediction[3,:,:]>=0.5
    ch5_prediction=torch_prediction[4,:,:]>=0.5
    clustermap[ch1_prediction>=0.5]=0
    clustermap[ch2_prediction>=0.5]=1
    clustermap[ch3_prediction>=0.5]=2
    clustermap[ch4_prediction>=0.5]=3
    clustermap[ch5_prediction>=0.5]=4
    clustermap_list.append(ch1_prediction)
    clustermap_list.append(ch2_prediction)
    clustermap_list.append(ch3_prediction)
    clustermap_list.append(ch4_prediction)
    clustermap_list.append(ch5_prediction)
    pg.plot_grids(clustermap_list,torch_prediction  )
    
    transformed_mip=img_norm2.squeeze(dim=0)
    transformed_mip=transformed_mip.squeeze(dim=0).cpu()
    print(transformed_mip.shape)
    #print(clustermap.shape)
    
    cropped_clustermap=cropped_transform((from_np(clustermap)).unsqueeze(dim=0))
    visualize.visualize(img_norm, )
    print(cropped_clustermap.shape)
    tifffile.imwrite(
        out_file_name_masks[i],
        cropped_clustermap.squeeze(dim=0).detach().cpu().numpy()
    )