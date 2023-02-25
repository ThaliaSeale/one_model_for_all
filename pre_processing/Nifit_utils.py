import nibabel as nib
import os
import numpy as np


def load_nifti(data_path, file_name):
    file_path = os.path.join(data_path, file_name)
    # print(file_path)
    img = nib.load(file_path)
    return img

def nifti_to_array(img):
    nifti_array = np.array(img.dataobj)
    return nifti_array

def save_arr(arr, save_path, file_name,aff):
    new_image = nib.Nifti1Image(arr, affine=aff)
    save_path = os.path.join(save_path,file_name)
    nib.save(new_image, save_path)

def merge_nifti(images):
    return nib.funcs.concat_images(images)

    
