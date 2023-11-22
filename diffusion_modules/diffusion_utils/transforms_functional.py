import torchvision.transforms.functional as TF

def adjust_sharpness_2(image):
    # Apply a functional transformation
    transformed_image = TF.adjust_sharpness(image, sharpness_factor=2.0)
    return transformed_image

def autocontrast(image):
    transformed_image = TF.autocontrast(image)
    return transformed_image

def hflip(image):
    transformed_image = TF.hflip(image)
    return transformed_image

def rotate(image,angle):
    transformed_image = TF.rotate(image, angle=angle,fill=0,interpolation=TF.InterpolationMode.BILINEAR)
    return transformed_image

def centercrop(image,output_size):
    transformed_image = TF.center_crop(image, output_size=output_size)
    return transformed_image

def zoomout(image,zoom_size):
    ot = TF.centercrop(image,output_size=zoom_size)
    transformed_image = TF.resize(ot, size=128)
    return transformed_image