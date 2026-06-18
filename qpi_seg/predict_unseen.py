#an improvised version of predict which predicts on completely unseen images :) and saves the masks
import torch
import tifffile
import os
import torchvision.transforms.v2 as transforms_v2
from models.unet import UNet
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import qpi_seg.visualize as visual_test
import qpi_seg.plot_grids as pg
import qpi_seg.split_mask_5_channels as split

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
assert torch.cuda.is_available()
model=UNet(depth=6,in_channels=1,out_channels=5, num_fmaps=32,final_activation=nn.Softmax()).to(device)
model_save=torch.load('/home/S-DC/ARTEMIS2D/checkpoints_unet1/checkpoint_epoch_190.pt')
model.load_state_dict(model_save['model_state_dict'])
model=model.to(device)

test_images_folder=r'/mnt/efs/dl_jrc/student_data/S-DC/MIP_stitched'
test_masks_folder=r'/mnt/efs/dl_jrc/student_data/S-DC/Mask_stitched'
test_masks_files=os.listdir(test_masks_folder)
#TODO: Export masks to be usable by Fiji
unet_masks_output_folder=r'/mnt/efs/dl_jrc/student_data/S-DC/Mask_stitched_output_unet'


test_files= os.listdir(test_images_folder)
test_images = [tifffile.imread(os.path.join(test_images_folder,file)) for file in test_files]

#for_calculating_dice
test_masks = [tifffile.imread(os.path.join(test_masks_folder,file)) for file in test_masks_files]


out_file_name_stems=[os.path.splitext(file)[0]+'_unet_masks.tiff'for file in test_files ]

out_file_name_masks=[os.path.join(unet_masks_output_folder, file) for file in out_file_name_stems]

from_np=transforms_v2.Lambda(lambda x: torch.from_numpy(x))
transform = transforms_v2.CenterCrop((832,832))


norm_min=13300
norm_max=14100
model.eval()
clustermaps=[]
for i in range(10,11):
    torch_test_mask=from_np(test_masks[i])
    torch_test_image = from_np(test_images[i])
    torch_test_mask=torch_test_mask.to(torch.int64)
    mask_norm=torch_test_mask.unsqueeze(dim=0).to(device)
    mask_norm2=mask_norm.unsqueeze(dim=0).to(device)
    
    torch_test_image=torch_test_image.float()
    img_norm= (torch_test_image - norm_min) / (norm_max - norm_min)
    img_norm=img_norm.unsqueeze(dim=0).to(device)
    img_norm2=img_norm.unsqueeze(dim=0).to(device)
    img_norm2=transform(img_norm2)
    mask_norm2=transform(mask_norm2).squeeze(dim=0)
    print(mask_norm2.shape)
    #split mask into 5 channels 
    channeled_mask=split.split_into_channels(mask_norm2)
    torch_prediction2=model(img_norm2) #prediction
    clustermap=np.zeros((832,832))
    torch_prediction=torch_prediction2.reshape(5,832,832).detach().cpu() #reshape to 5 channels and detach and cpu
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
    
    

    #pad clustermap with 4 pixels 
    #padded_clustermap=np.pad(clustermap,((2,2),(2,2)),'constant',constant_values=(0,0))
    #clustermaps.append(padded_clustermap)

#for i in range(len(out_file_name_masks)):
#    tifffile.imwrite(
#        out_file_name_masks[i],
#        clustermaps[i]
#    )
channeled_mask_list=[]
channeled_mask_list.append(channeled_mask.detach().to('cpu')[0,:,:])
channeled_mask_list.append(channeled_mask.detach().to('cpu')[1,:,:])
channeled_mask_list.append(channeled_mask.detach().to('cpu')[2,:,:])
channeled_mask_list.append(channeled_mask.detach().to('cpu')[3,:,:])
channeled_mask_list.append(channeled_mask.detach().to('cpu')[4,:,:])

#oncat_channel1_reshaped=concat_channel1_images.unsqueeze(dim=1)
pg.plot_grids(channeled_mask_list,torch_prediction  )

class multiclassDiceCoefficient(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    dice_coeffs=list()
    def forward(self, prediction, target):
        dice_coeffs=list()
        for i in range(5):
            numerator = 2 * (prediction[:,i,:,:] * target[:,i,:,:]).sum()
            denominator = (prediction[:,i,:,:]** 2).sum() + (target[:,i,:,:] ** 2).sum()
            dice_coeffs.append(numerator / denominator.clamp(min=self.eps))
        return dice_coeffs

channeled_mask_unsqueeze=channeled_mask.unsqueeze(dim=0)

dice=multiclassDiceCoefficient()
dice_list = dice(torch_prediction2,channeled_mask_unsqueeze)
print(dice_list)
#visual_test.visualize(img_norm2.cpu().reshape(832,832), mask_norm2.cpu().reshape(832,832), clustermap)


#visual_test.visualize(test_image, test_mask,padded_clustermap)
#plt.savefig('test_predict10.png')

#the targets here are to export a merged mask of 5 channels


