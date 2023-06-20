import tifffile
import numpy as np

def save_as_tiff(images, tiff_filename):
    assert len(images.shape) == 4, "Shape of images should be [N,C,H,W]"
    transposed_images = np.transpose(images.cpu().numpy(), (0, 2, 3, 1))
    image_uint8 = (transposed_images*255).astype(np.uint8)
    tifffile.imwrite(tiff_filename, image_uint8, photometric='minisblack')