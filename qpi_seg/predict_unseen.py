##UNET TESTING SCRIPT- export a merged mask of 5 channels on completely unseen images :) 
import torch
import tifffile
import os
import torchvision.transforms.v2 as transforms_v2
from models.unet import UNet
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import qpi_seg.plot_grids as pg
import qpi_seg.split_mask_5_channels as split
import qpi_seg.visualize_unseen_unmasked as visualize

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#assert torch.cuda.is_available()
model=UNet(depth=6,in_channels=1,out_channels=5, num_fmaps=32,final_activation=nn.Softmax()).to(device)

#change the model path
model_path=r"C:\Users\BarmanLab\Downloads\checkpoint_epoch_190.pt"
#add map_location if using on CPU
#model_save=torch.load(model_path,map_location=torch.device('cpu'))

#if using GPU uncomment the following two lines
assert torch.cuda.is_available()
model_save=torch.load(model_path)

model.load_state_dict(model_save['model_state_dict'])
model=model.to(device)

#put your unseen images here
test_images_folder=r'D:\TRAINING_DATA_FINAL\TEST_MIP'

#put folder where masks will be saved
unet_masks_output_folder=r'D:\TRAINING_DATA_FINAL\TEST_MASK'


test_files= os.listdir(test_images_folder)
test_images = [tifffile.imread(os.path.join(test_images_folder,file)) for file in test_files]
#print(test_images[0].shape)

out_file_name_stems=[os.path.splitext(file)[0][:30]+'_unet_masks.tiff'for file in test_files ]

out_file_name_masks=[os.path.join(unet_masks_output_folder, file) for file in out_file_name_stems]

from_np=transforms_v2.Lambda(lambda x: torch.from_numpy(x))

im_size=2016
transform = transforms_v2.CenterCrop((im_size,im_size))

#for victor's dataset may have to play with this
norm_min=13300
norm_max=14100
model.eval()
model = model.to(device)
clustermaps=[]


for i in range(0,1):
    #w=test_images[i].shape[0]
    #h=test_images[i].shape[1]
    torch_test_image = from_np(test_images[i])
    torch_test_image=torch_test_image.float()
    img_norm= (torch_test_image - norm_min) / (norm_max - norm_min)
    img_norm=img_norm.unsqueeze(dim=0).to(device)
    img_norm2=transform(img_norm.unsqueeze(dim=0).to(device))
    #TODO: for future extensions - save size of the image
    torch_prediction=model(img_norm2)
    clustermap=np.zeros((im_size,im_size))
    torch_prediction=torch_prediction.reshape(5,im_size,im_size).detach().cpu()
    ch1_prediction = torch_prediction[0,:,:]
    ch2_prediction = torch_prediction[1,:,:]
    ch3_prediction= torch_prediction[2,:,:]
    ch4_prediction=torch_prediction[3,:,:]
    ch5_prediction=torch_prediction[4,:,:]

    clustermap[ch1_prediction>=0.5]=0
    clustermap[ch2_prediction>=0.5]=1
    clustermap[ch3_prediction>=0.5]=2
    clustermap[ch4_prediction>=0.5]=3
    clustermap[ch5_prediction>=0.5]=4
    
    transformed_mip=img_norm2.squeeze(dim=0)
    transformed_mip=transformed_mip.squeeze(dim=0).cpu()
    print(transformed_mip.shape)
    #print(clustermap.shape)
    visualize.visualize(transformed_mip, clustermap)
    #clustermaps.append(clustermap)

#for i in range(len(out_file_name_masks)):
#    tifffile.imwrite(
#        out_file_name_masks[i],
#        clustermaps[i]
#    )

#plot grid is a function that outputs 10 images (5 x 2 pattern)
#pg.plot_grids(test_images[0:5],clustermaps[0:5] )




