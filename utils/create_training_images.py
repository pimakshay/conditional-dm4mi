import tifffile
import numpy as np
import h5py
import argparse
import SimpleITK as sitk
import os
# parser = argparse.ArgumentParser()
# parser.add_argument('--image', required=True, help='path to image')
# parser.add_argument('--name', required=True, help='name of dataset')
# parser.add_argument('--edgelength', type=int, default=128, help='input batch size')
# parser.add_argument('--stride', type=int, default=32, help='the height / width of the input image to network')
# parser.add_argument('--target_dir', required=True, help='path to store training images')

# opt = parser.parse_args()
# print(opt)

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        # Create the folder
        os.makedirs(folder_path)
        print("New Folder created successfully: ", folder_path)
    else:
        print("Folder already exists.")

def rescale_image(image):
    filter = sitk.RescaleIntensityImageFilter()
    filter.SetOutputMaximum(255)
    filter.SetOutputMinimum(0)
    rescaled_img = filter.Execute(image)
    return rescaled_img

def save_as_hdf5(inputpath="", datasetname="", targetdir="", image_size=64, stride=32, batch_size=1):
    create_folder(targetdir)
    img = sitk.GetArrayFromImage(rescale_image(sitk.ReadImage(str(inputpath)))) #tifffile.imread(str(opt.image))

    count = 0

    edge_length = image_size #image dimensions
    stride = stride #stride at which images are extracted

    N = batch_size #edge_length
    M = edge_length
    O = edge_length

    I_inc = stride
    J_inc = stride
    K_inc = stride

    target_direc = str(targetdir)
    count = 0
    for i in range(0, img.shape[0], I_inc):
        for j in range(0, img.shape[1], J_inc):
            for k in range(0, img.shape[2], K_inc):
                subset = img[i:i+N, j:j+M, k:k+O]
                if subset.shape == (N, M, O):
                    hdf5_file = target_direc+"/"+str(datasetname)+"_"+str(count)+".hdf5"
                    f = h5py.File(hdf5_file, "w")
                    f.create_dataset('data', data=subset, dtype="i8", compression="gzip")
                    f.close()
                    count += 1
    print("Total volumes created: ", count)
    
# save_as_hdf5(inputpath=opt.image, datasetname=opt.name, targetdir=opt.target_dir, image_size=opt.edgelength, stride=opt.stride)