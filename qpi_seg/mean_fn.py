import torchvision.transforms.v2 as transforms_v2
import torch
import tifffile
from tqdm import tqdm
import os
def mean_fn(image_folder, image_files):
    num_samples=len(image_files)
    loaded_imgs = [None]*num_samples
    count = num_samples*836*836 # TODO: make bAsed on width and heigth
    from_np = transforms_v2.Lambda(lambda x: torch.from_numpy(x)) #TODO change name
    sum_of_dataset=torch.tensor(0.0)
    sum_sq_of_dataset=torch.tensor(0.0)
    for sample_ind in tqdm(range(num_samples),desc="Reading images"):
        img_path = os.path.join(image_folder, image_files[sample_ind])
        image = from_np(tifffile.imread(img_path))
        image_float=image.float()
        sum_of_dataset += image_float.sum(axis=[0,1])
        sq_dataset= torch.square(image_float)
        sum_sq_of_dataset += sq_dataset.sum(axis=[0,1])
    mean = sum_of_dataset/count
    var = (sum_sq_of_dataset/count)- (mean**2)
    std = torch.sqrt(var)
    return mean, std