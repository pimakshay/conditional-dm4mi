import torch
from skimage.util import random_noise

class AddNoise(object):
    def __init__(self, noise_type):
        self.noise_type = noise_type

    def __call__(self, sample):
        image = sample

        noisy_image = add_noise(image, self.noise_type)
        
        return noisy_image

def add_noise(image, noise_type="gaussian"):
    """
    Add noise to an image tensor using the specified noise type.

    Args:
        image (torch.Tensor): Input image tensor of shape (C, H, W).
        noise_type (str): Type of noise to be applied. Options: 'gaussian', 'salt', 'pepper', 's&p', 'speckle'.

    Returns:
        torch.Tensor: Image tensor with added noise.
    """
    image_np = image.numpy()  # Convert the image tensor to a NumPy array

    if noise_type == 'gaussian':
        noisy_image = random_noise(image_np, mode='gaussian')
    elif noise_type == 'salt':
        noisy_image = random_noise(image_np, mode='salt')
    elif noise_type == 'pepper':
        noisy_image = random_noise(image_np, mode='pepper')
    elif noise_type == 's&p':
        noisy_image = random_noise(image_np, mode='s&p')
    elif noise_type == 'speckle':
        noisy_image = random_noise(image_np, mode='speckle')
    else:
        raise ValueError("Invalid noise type. Options: 'gaussian', 'salt', 'pepper', 's&p', 'speckle'.")

    noisy_image = torch.from_numpy(noisy_image)  # Convert the NumPy array back to a tensor

    return noisy_image

def join_noise_prior_dataset(noisy_dataloader, prior_dataloader):
    # Iterate over the dataloaders and join corresponding items
    joined_dataset = []
    for noisy_image, prior_image in zip(noisy_dataloader, prior_dataloader):
        joined_item = {
            'x_cond': noisy_image[0],
            'x_prior': prior_image[0]
        }
        joined_dataset.append(joined_item)

    # Create a new dataset for the joined data
    return joined_dataset


def get_denoise_dataset(dataloader, batch_size):
    # Create a new dataset with noisy images
    noisy_dataset = []

    # Iterate over each image in the original dataloader and apply the noise transform
    for images, labels in dataloader():
        noisy_images = [add_noise(image, noise_type="gaussian") for image in images]
        noisy_dataset.extend(list(zip(noisy_images, labels)))

    # Create a new dataloader using the noisy dataset
    noisy_dataloader = torch.utils.data.DataLoader(noisy_dataset, batch_size=batch_size, shuffle=False)

    return join_noise_prior_dataset(noisy_dataloader, dataloader)