import nibabel as nib
import os
from nibabel.testing import data_path
import numpy as np
import numpy.ma as ma

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


if __name__ == "__main__":

    label_path = "/home/sedm6251/projectMaterial/skullStripped/ATLAS/labels"
    mask_path = "/home/sedm6251/projectMaterial/skullStripped/ATLAS/masks"
    img_path = "/home/sedm6251/projectMaterial/skullStripped/ATLAS/normed_images"
    save_path = "/home/sedm6251/projectMaterial/skullStripped/ATLAS/trimmed_labels_ints"
    load_trimmed_label_path = "/home/sedm6251/projectMaterial/skullStripped/ATLAS/trimmed_labels_ints"

    cases = os.listdir(label_path)
    cases.sort()
    # cases = ["sub-r039s002_label.nii.gz"]
    # cases = ["sub-r001s022_label.nii.gz","sub-r001s023_label.nii.gz"]

    for case in cases:
        
        # print(case)
        case_name_no_suffix = case[:case.rfind("_label.nii.gz")]
        # print(case_name_no_suffix)
        img_name = case_name_no_suffix + "_normed.nii.gz"
        label_name = case_name_no_suffix + "_label.nii.gz"
        load_trimmed_label_name = case_name_no_suffix + "_label_trimmed.nii.gz"
        save_name = case_name_no_suffix + "_label_trimmed.nii.gz"
        mask_name = case_name_no_suffix + "_mask.nii.gz"



        img_file = load_nifti(img_path,img_name)
        label_file = load_nifti(label_path,label_name)
        load_trimmed_label_file = load_nifti(load_trimmed_label_path,load_trimmed_label_name)
        # mask_file = load_nifti(mask_path,mask_name)

        img_array = nifti_to_array(img_file)
        # label_array = nifti_to_array(label_file)
        load_trimmed_label_array = nifti_to_array(load_trimmed_label_file)
        # mask_array = nifti_to_array(mask_file)

        # trimmed_label = np.multiply(label_array,mask_array)

        # print("Image datatype: ", img_array.dtype)
        # print("Mask datatype: ", mask_array.dtype)
        # print("Loaded Trimmed label datatype: ", load_trimmed_label_array.dtype)
        # print("Image Mean: ", np.mean(img_array))
        # print("Img Std: ", np.std(img_array))

        if img_array.dtype != np.float32:
            print("Image not float 32. Case:")
            print(case)
        if load_trimmed_label_array.dtype != np.uint8:
            print("Trimmed label not int 8. Case:")
            print(case)
            print("Datatype:")
            print(load_trimmed_label_array.dtype)

        # trimmed_label = np.multiply(trimmed_label,100)

        # print(case)
        # trimmed_label = load_trimmed_label_array.round(0).astype(np.uint8)

        # print(np.max(load_trimmed_label_array))
        # print(np.max(load_trimmed_label_array))
        if np.max(load_trimmed_label_array) != 1:
            print("original label max not 1, value:, case:")
            print(np.max(load_trimmed_label_array))
            print(case)

        # if np.max(trimmed_label) != 1.0:
        #     print("trimmed label max not 1, case:")
        #     print(case)

        if np.min(load_trimmed_label_array) != 0:
            print("original label min not 0, case:")
            print(case)

        # if np.min(trimmed_label) != 0.0:
        #     print("trimmed label min not 0, case:")
        #     print(case)

        # save_arr(trimmed_label,save_path,save_name,label_file.affine)
