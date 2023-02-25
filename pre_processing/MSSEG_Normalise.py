import nibabel as nib
from nibabel import affines
import os
from nibabel.testing import data_path
import numpy as np
import numpy.ma as ma
from glob import glob
import shutil

def load_nifti(data_path, file_name):
    file_path = os.path.join(data_path, file_name)
    print(file_path)
    img = nib.load(file_path)
    return img

def nifti_to_array(img):
    nifti_array = np.array(img.dataobj)
    return nifti_array

def normalise(arr, mask):
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

def crop_center(img,cropx,cropy,cropz):
    x,y,z = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    startz = z//2-(cropz//2)  


    # removed_1 = img[:startx,:,:]
    # removed_2 = img[:,:starty,:]
    # removed_3 = img[:,:,:startz]
    # removed_4 = img[-startx:,:,:]
    # removed_5 = img[:,-starty:,:]
    # removed_6 = img[:,:,-startz:]

    # if np.any(removed_1==1):
    #     print("Removed 1")
    # elif np.any(removed_2==1):
    #     print("Removed 2")
    # elif np.any(removed_3==1):
    #     print("Removed 3")
    # elif np.any(removed_4==1):
    #     print("Removed 4")
    # elif np.any(removed_5==1):
    #     print("Removed 5")
    # elif np.any(removed_6==1):
    #     print("Removed 6")
    cropped = img.slicer[startx:startx+cropx,starty:starty+cropy, startz:startz+cropz]
    return cropped

if __name__ == "__main__":


    training_set_path = "/home/shared_space/data/msseg_2016/preprocessed_dm2"
    # training_set_path = "/home/sedm6251/MSSEG_TEST"


    subsets = os.listdir(training_set_path)
    subsets.sort()
    # print("Found subsets:")
    # print(subsets)
    for subset in subsets:
        # e.g. subsets = [R001, R002,...]
        if not os.path.isdir(os.path.join(training_set_path, subset)):
            # print("Subset path is not a directory")
            continue  # In case of subfiles instead of subfolders, e.g. readme etc.
        
        # subset path is e.g. /home/shared_space/data/ATLAS_R2.0/ATLAS_2/Training/R001/
        subset_path = os.path.join(training_set_path, subset)
        modality_folders = ["channel_FLAIR", "channel_T1", "channel_GADO", "channel_T2", "channel_DP"]
        # modality_folders = os.path.join(subset_path,"channel*")
        # print("Found cases:")
        # print(cases)
        print(subset)

        modalities_array = []
        affine_array = []       
        normed_array = [] 
        for modality_folder in modality_folders:
            if not os.path.isdir(os.path.join(subset_path, modality_folder)):
                continue  # In case of subfiles instead of subfolders, e.g. readme etc.

            modality_file = glob(os.path.join(subset_path,modality_folder,"*.nii.gz"))
            # print(modality_file)
            modality_nifti = nib.load(modality_file[0])
            modality_arr = nifti_to_array(modality_nifti)

            # print(modality_nifti.header.get_zooms())

            # modality_arr = affines.apply_affine(modality_nifti.affine,modality_arr)
            
            modality_mask = modality_arr != 0
            modality_mask = modality_mask.astype(np.bool)
            modality_normed = normalise(modality_arr, modality_mask)


            normed_array.append(modality_normed)
            modalities_array.append(modality_nifti)
            # affine_array.append(modality_nifti.affine)
        # concat_modalities = nib.funcs.concat_images(,check_affines=False)

        concat_modalities = np.stack(normed_array, axis = -1)

        seg_file = nib.load(os.path.join(subset_path,"ground_truth/Consensus_int8.nii.gz"))
        seg_arr = nifti_to_array(seg_file)
        # print(seg_arr.shape)
        # print(seg_arr[4:-4, 30:-30, 30:-30].shape)
        # cropped_seg = crop_center(seg_file,150,200,200)
        # cropped_seg_arr = nifti_to_array(cropped_seg)
        
        # print("Cropped: ", cropped_seg.shape)
        

        # print(concat_modalities.shape)
        # print(seg_arr.shape)

        save_name = subset + ".nii.gz"

        # label_path = os.path.join(subset_path,"ground_truth/Consensus.nii.gz")

        label_save_path = os.path.join("/home/sedm6251/projectMaterial/datasets/MSSEG_2016/Labels", save_name)
        file_save_path = os.path.join("/home/sedm6251/projectMaterial/datasets/MSSEG_2016/Normed",save_name)

        # new_image = nib.Nifti1Image(concat_modalities, affine=modalities_array[0].affine)
        # new_image = nib.Nifti1Image(concat_modalities, affine=None)
        # cropped_seg = nib.Nifti1Image(cropped_seg_arr,affine=new_image.affine)
        
        # print(cropped_seg.header.get_zooms())
        # print(new_image.header.get_zooms())
        # nib.save(cropped_seg,label_save_path)
        # nib.save(new_image, file_save_path)

        # save_arr(seg_arr)

        # seg_path = os.path.join(subset_path,"ground_truth/Consensus_int8.nii.gz")
        # shutil.copy(seg_path,label_save_path)


        
        


        img_file = nib.Nifti1Image(concat_modalities, affine=None)
        seg_file_from_arr = nib.Nifti1Image(seg_arr, affine=None)
        
        seg_file_from_arr.header.set_sform(np.diag([-1, -1, 1, 1]), code='aligned')
        img_file.header.set_sform(np.diag([-1, -1, 1, 1]), code='aligned')
        
        nib.save(img_file,file_save_path)
        nib.save(seg_file_from_arr,label_save_path)

        # save_arr(seg_arr, "/home/sedm6251/projectMaterial/datasets/MSSEG_2016/Labels", save_name,seg_file.affine)
        # save_arr(concat_modalities, "/home/sedm6251/projectMaterial/datasets/MSSEG_2016/Normed", save_name, seg_file.affine)
        # save_arr(concat_modalities, "/home/sedm6251/projectMaterial/datasets/MSSEG_2016/Normed", save_name, None)

        print("")
        
        