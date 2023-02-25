import nibabel as nib
import os
from nibabel.testing import data_path
import numpy as np
import numpy.ma as ma

def load_nifti(data_path, file_name):
    file_path = os.path.join(data_path, file_name)
    print(file_path)
    img = nib.load(file_path)
    return img

def nifti_to_array(img):
    nifti_array = np.array(img.dataobj)
    return nifti_array

def normalise(arr,mask):
    # print(mask.shape)
    m = ~mask
    masked_arr = ma.masked_array(arr, mask = m)
    mean = masked_arr.mean()
    std = masked_arr.std()

    normed = (arr - mean)/std

    return normed

def save_arr(arr, save_path, file_name,aff):
    new_image = nib.Nifti1Image(arr, affine=aff)
    save_path = os.path.join(save_path,file_name)
    nib.save(new_image, save_path)

if __name__ == "__main__":

    # data_path = "/home/sedm6251/projectMaterial/skullStripped/ATLAS/images"
    # save_path = "/home/sedm6251/projectMaterial/skullStripped/ATLAS/normed_images"
    # mask_path = "/home/sedm6251/projectMaterial/skullStripped/ATLAS/masks"
    
    # brats_d_path = "/home/shared_space/data/BRATS_Decathlon_2016_17/BRATS_Normalised_with_brainmask/normed"
    brats_d_path = "/home/shared_space/data/BRATS_Decathlon_2016_17/BRATS_merged_labels_inc_edema"
    brats_nifti_file = load_nifti(data_path=brats_d_path, file_name="BRATS_001merged.nii.gz")
    brats_nifti_array = nifti_to_array(brats_nifti_file)
    print(brats_nifti_array.dtype)
    


    cases = os.listdir(data_path)
    cases.sort()

    for case in cases[-196:]:
        
        # print(case)
        case_name_no_suffix = case[:case.rfind("_img.nii.gz")]
        print(case_name_no_suffix)

        save_name = case_name_no_suffix + "_normed.nii.gz"
        mask_name = case_name_no_suffix + "_mask.nii.gz"

        mask_file = load_nifti(data_path=mask_path,file_name=mask_name)
        brain_mask_array = nifti_to_array(mask_file)
        brain_mask_array = brain_mask_array.astype(np.bool)


        nifti_file = load_nifti(data_path=data_path, file_name=case)
        image_affine = nifti_file.affine
        nifti_array = nifti_to_array(nifti_file)

        normed = normalise(nifti_array,brain_mask_array)
        # print(normed.shape)
        save_arr(normed, save_path, save_name,image_affine)


    
    

    
