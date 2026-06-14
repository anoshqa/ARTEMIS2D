import torchvision.transforms.v2 as transforms_v2
import torch
import tifffile
from tqdm import tqdm
import os
def mean_fn(image_folder, image_files):
    num_samples=len(image_files)
    loaded_imgs = [None]*num_samples
    from_np = transforms_v2.Lambda(lambda x: torch.from_numpy(x))
    sum_of_dataset=torch.tensor(0.0)
    sum_sq_of_dataset=torch.tensor(0.0)
    for sample_ind in tqdm(range(num_samples),desc="Reading images"):
        img_path = os.path.join(image_folder, image_files[sample_ind])
        image = from_np(tifffile.imread(img_path))
        sum_of_dataset += image.sum(axis=[0,1])
        sq_dataset= torch.square(image)
        sum_sq_of_dataset += sq_dataset.sum(axis=[0,1])
    mean = sum_of_dataset/(num_samples)
    std = torch.sqrt((sum_sq_of_dataset/num_samples)- mean**2)
    return mean, std