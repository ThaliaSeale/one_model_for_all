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
    img_save_path = "/home/sedm6251/projectMaterial/skullStripped/ATLAS/cropped/img"
    label_save_path = "/home/sedm6251/projectMaterial/skullStripped/ATLAS/cropped/label"
    load_trimmed_label_path = "/home/sedm6251/projectMaterial/skullStripped/ATLAS/trimmed_labels_ints"

    cases = os.listdir(mask_path)
    cases.sort()
    # cases = ["sub-r039s002_label.nii.gz"]
    # cases = ["sub-r001s022_label.nii.gz","sub-r001s023_label.nii.gz"]
    dim_0_min = 1000
    dim_1_min = 1000
    dim_2_min = 1000

    dim_0_max = 0
    dim_1_max = 0
    dim_2_max = 0
    length = str(len(cases))
    i = 1
    for case in cases:
        i+=1
        print(str(i) + "/" + length)
        print(case)
        case_name_no_suffix = case[:case.rfind("_mask.nii.gz")]
        # print(case_name_no_suffix)
        img_name = case_name_no_suffix + "_normed.nii.gz"
        label_name = case_name_no_suffix + "_label.nii.gz"
        load_trimmed_label_name = case_name_no_suffix + "_label_trimmed.nii.gz"
        img_save_name = case_name_no_suffix + "_img_cropped.nii.gz"
        label_save_name = case_name_no_suffix + "_label_cropped.nii.gz"
        mask_name = case_name_no_suffix + "_mask.nii.gz"

        img_file = load_nifti(img_path,img_name)
        # label_file = load_nifti(label_path,label_name)
        load_trimmed_label_file = load_nifti(load_trimmed_label_path,load_trimmed_label_name)
        mask_file = load_nifti(mask_path,mask_name)

        img_array = nifti_to_array(img_file)
        # label_array = nifti_to_array(label_file)
        load_trimmed_label_array = nifti_to_array(load_trimmed_label_file)
        mask_array = nifti_to_array(mask_file)
        # print(mask_array.shape)
        mask_args = np.argwhere(mask_array)

        # dim_0_min = min(np.min(mask_args[:,0]),dim_0_min)
        # dim_1_min = min(np.min(mask_args[:,1]),dim_1_min)
        # dim_2_min = min(np.min(mask_args[:,2]),dim_2_min)

        # dim_0_max = max(np.max(mask_args[:,0]),dim_0_max)
        # dim_1_max = max(np.max(mask_args[:,1]),dim_1_max)
        # dim_2_max = max(np.max(mask_args[:,2]),dim_2_max)

        dim_0_min = np.min(mask_args[:,0],axis=0)
        dim_1_min = np.min(mask_args[:,1],axis=0)
        dim_2_min = np.min(mask_args[:,2],axis=0)

        dim_0_max = np.max(mask_args[:,0],axis=0)
        dim_1_max = np.max(mask_args[:,1],axis=0)
        dim_2_max = np.max(mask_args[:,2],axis=0)
        if dim_0_max - dim_0_min < 128:
            print("dim 0 fails")
            break
        if dim_1_max - dim_1_min < 128:
            print("dim 1 fails")
            break
        if dim_2_max - dim_2_min < 128:
            print("dim 2 fails")
            break

        cropped_img = img_array[dim_0_min : dim_0_max, dim_1_min : dim_1_max, dim_2_min : dim_2_max]
        cropped_label = load_trimmed_label_array[dim_0_min : dim_0_max, dim_1_min : dim_1_max, dim_2_min : dim_2_max]

        save_arr(cropped_img, img_save_path, img_save_name, img_file.affine)
        save_arr(cropped_label, label_save_path, label_save_name, load_trimmed_label_file.affine)


    # print(dim_0_min)
    # print(dim_1_min)
    # print(dim_2_min)

    # print(dim_0_max)
    # print(dim_1_max)
    # print(dim_2_max)


