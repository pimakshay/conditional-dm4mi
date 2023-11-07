import torchvision, torch
from torchvision import transforms 
from torchvision.transforms import Compose
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import random_split
from diffusion_modules.diffusion_utils import dataloaderDenoise
from diffusion_modules.diffusion_utils.hdf5dataset import HDF5Dataset
from diffusion_modules.diffusion_utils.pddpm_dataloader import TrainDataModule, TrainIXIDataModule, get_all_test_dataloaders

# synthrad dataset
def load_brain_hdf5(image_dir, noise_type="gaussian", dynamic_range=255):
    dataset = HDF5Dataset(image_dir=image_dir, input_transform=transforms.Compose([
#                               transforms.ToTensor()
                              ]), dynamic_range=dynamic_range)
    noisy_dataset = HDF5Dataset(image_dir=image_dir, input_transform=transforms.Compose([
#                               transforms.ToTensor(),
                              dataloaderDenoise.AddNoise("gaussian")
                              ]), dynamic_range=dynamic_range)    
    return dataset, noisy_dataset

# fastmri + ixi dataset
def load_fastmri(image_dir, noise_type="gaussian", variance=0.01, dynamic_range=255, batch_size=32, target_size=128):
    train_data_module = TrainDataModule(
        split_dir=image_dir,
        target_size=target_size,
        batch_size=batch_size,
        input_transform=transforms.Compose([
#                               transforms.ToTensor(),
                              dataloaderDenoise.AddNoise("gaussian", var=variance)
                              ]), 
        )    

    train_dataset = train_data_module.train_dataloader().dataset
    noisy_dataset = train_data_module.noisy_train_dataloader().dataset
#     val_dataset = train_data_module.val_dataloader().dataset

    return train_dataset, noisy_dataset


# ixi dataset
def load_ixi(image_dir, noise_type="gaussian", variance=0.01, dynamic_range=255, batch_size=32, target_size=128):
    train_data_module = TrainIXIDataModule(
        split_dir=image_dir,
        target_size=target_size,
        batch_size=batch_size,
        input_transform=transforms.Compose([
#                               transforms.ToTensor(),
                              dataloaderDenoise.AddNoise("gaussian", var=variance)
                              ]), 
        )    

    train_dataset = train_data_module.train_dataloader().dataset
    noisy_dataset = train_data_module.noisy_train_dataloader().dataset
#     val_dataset = train_data_module.val_dataloader().dataset

    return train_dataset, noisy_dataset


def load_ixi_sr(image_dir, image_size, scale_factor=1, dynamic_range=255, batch_size=32, target_size=128):
    lr_data_transform = transforms.Compose([
        transforms.Resize(int(image_size/scale_factor)), transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
#         transforms.ToTensor(), # Scales data into [0,1] 
    ])     
    train_data_module = TrainIXIDataModule(
        split_dir=image_dir,
        target_size=target_size,
        batch_size=batch_size,
        input_SR_transform=lr_data_transform
        )    

    train_dataset = train_data_module.train_dataloader().dataset
    lr_dataset = train_data_module.lr_train_dataloader().dataset
#     val_dataset = train_data_module.val_dataloader().dataset

    return train_dataset, lr_dataset

def get_conditioned_dataloader(dataloader, cond):
    """
    gets conditioned dataloader

    Args:
        image (torch.Tensor): Input image tensor of shape (C, H, W).
        noise_type (str): Type of noise to be applied. Options: 'gaussian', 'salt', 'pepper', 's&p', 'speckle'.

    Returns:
        torch.Tensor: Image tensor with added noise.
    """    
    if cond=="SR":
        conditioned_dataloader = None #callSR()
    elif cond=="denoise":
        conditioned_dataloader = dataloaderDenoise.get_denoise_dataset(dataloader)
    
    return conditioned_dataloader

def show_images(dataset, num_samples=24, cols=4, random=True, save_all=False, filename="abc.png"):
    """ Plot400s some samples from the dataset """
    random_img_idx = np.random.randint(0, high=len(dataset), size=len(dataset), dtype=int)
    rows = int(num_samples//cols)
    img_size = dataset[0].shape[1]
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(14,14))    
    img_count=0
    for i in range(rows):
        for j in range(cols):
            random_index = random_img_idx[i*cols + j] if random else img_count#int(np.random.random()*len(dataset))
            img = dataset[random_index] # use for stanford_Cars
            out = img[0].numpy().reshape(img_size, img_size)
            axes[i, j].imshow(out, cmap='Greys')    
            img_count = img_count + 1    
    if save_all:
        assert filename is not None, "Filename missing.."
        plt.savefig(filename)

class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, cond_dataset, prior_dataset):
        self.cond_dataset = cond_dataset
        self.prior_dataset = prior_dataset

    def __getitem__(self, index):
        cond_image = self.cond_dataset[index]
        prior_image = self.prior_dataset[index]
        image = {'x_cond': cond_image,
                'x_prior': prior_image}
        return image
    
    def __len__(self):
        return min(len(self.cond_dataset), len(self.prior_dataset))

def split_dataset(dataset, tvt_ratio=[0.8,0.1,0.1]):
    dataset_size = len(dataset)
    # Define the sizes for train, validation, and test sets
    train_size = int(tvt_ratio[0] * dataset_size)  # 80% for training
    val_size = int(tvt_ratio[1] * dataset_size)   # 10% for validation
    test_size = dataset_size - train_size - val_size  # Remaining 10% for testing

    # Define the indices for train, validation, and test subsets
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, dataset_size))

    # Create Subset objects for the train, validation, and test subsets using the defined indices
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
        
    return train_dataset, val_dataset, test_dataset 