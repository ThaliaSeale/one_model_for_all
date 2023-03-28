import os
import shutil
import nibabel as nib
import numpy as np
import numpy.ma as ma
import torch
import matplotlib.pyplot as plt


cwd = os.getcwd()

with open(os.path.join(cwd,"pre_processing/pre_proc_TBI/channels_flair.txt")) as FLAIR_files:
    FLAIR_lines = [line.strip() for line in FLAIR_files]

with open(os.path.join(cwd,"pre_processing/pre_proc_TBI/channels_swige.txt")) as SWI_files:
    SWI_lines = [line.strip() for line in SWI_files]

with open(os.path.join(cwd,"pre_processing/pre_proc_TBI/channels_t1w.txt")) as T1_files:
    T1_lines = [line.strip() for line in T1_files]

with open(os.path.join(cwd,"pre_processing/pre_proc_TBI/channels_t2.txt")) as T2_files:
    T2_lines = [line.strip() for line in T2_files]

with open(os.path.join(cwd,"pre_processing/pre_proc_TBI/gtLabels_FLAIR_mc8cNoSEdhJune19.txt")) as labels_FLAIR_files:
    labels_FLAIR_lines = [line.strip() for line in labels_FLAIR_files]

with open(os.path.join(cwd,"pre_processing/pre_proc_TBI/gtLabels_SWIGE_mc8cNoSEdhJune19.txt")) as labels_SWI_files:
    labels_SWI_lines = [line.strip() for line in labels_SWI_files]

with open(os.path.join(cwd,"pre_processing/pre_proc_TBI/gtLabels_AllOld_mc8cNoSEdhJune19.txt")) as labels_merged_files:
    labels_merged_lines = [line.strip() for line in labels_merged_files]

with open(os.path.join(cwd,"pre_processing/pre_proc_TBI/roimasks.txt")) as brain_mask_files:
    brain_mask_lines = [line.strip() for line in brain_mask_files]

def skullstrip_on_mask(img: np.array, mask: np.array):
    corners = []
    corners.append(img[0,0,0])
    corners.append(img[-1,0,0])
    corners.append(img[0,-1,0])
    corners.append(img[0,0,-1])
    corners.append(img[-1,-1,0])
    corners.append(img[-1,0,-1])
    corners.append(img[0,-1,-1])
    corners.append(img[-1,-1,-1])

    avg_corner = np.average(corners)

    img[mask<0.5] = avg_corner
    return img

def normalise(arr,mask):
    # print(mask.shape)
    m = mask < 0.5
    masked_arr = ma.masked_array(arr, mask = m)
    mean = masked_arr.mean()
    std = masked_arr.std()

    normed = (arr - mean)/std
    # plt.imshow(mask)
    # plt.show()
    # plt.imshow(normed)
    # plt.show()

    return normed

def merge_labels(label_arr):
    merged = (label_arr > 0.5)
    merged = np.array(merged, dtype=np.uint8)
    return merged

def save_arr_to_nifti(arr, save_path, affine):
    nifti_img = nib.Nifti1Image(arr, affine)
    sform = np.diag([1, 1, 1, 1])
    # t = [-98,-134,-72,1]
    # sform[:,3] = t
    nifti_img.header.set_sform(sform,code="aligned")
    
    nib.save(nifti_img, save_path)

def get_uid(file_path: str):
    if "trio" in file_path:
        start_idx = file_path.find("trio/") + 5
        uid = file_path[start_idx : file_path.find("/",start_idx)]
    elif "verio" in file_path:
        start_idx = file_path.find("verio/") + 6
        uid = file_path[start_idx : file_path.find("/",start_idx)]
    elif "Norman" in file_path:
        start_idx = file_path.find("Norman/") + 7
        uid = file_path[start_idx : file_path.find("/",start_idx)]
    elif "Poe" in file_path:
        start_idx = file_path.find("Poe/") + 4
        uid = file_path[start_idx : file_path.find("/",start_idx)]
    elif "CENTER-TBI-2020-6" in file_path:
        start_idx = file_path.find("CENTER-TBI-2020-6/") + 18
        uid = file_path[start_idx : file_path.find("/",start_idx)]
    return uid

def create_save_name(file_path: str):
    uid = get_uid(file_path)
    if "trio" in file_path:
        start_idx = file_path.find("trio/") + 5
        save_name = "trio_" + uid + ".nii.gz"
    elif "verio" in file_path:
        start_idx = file_path.find("verio/") + 6
        save_name = "verio_" + uid + ".nii.gz"
    elif "Norman" in file_path:
        start_idx = file_path.find("Norman/") + 7
        save_name = "Norman_" + uid + ".nii.gz"
    elif "Poe" in file_path:
        start_idx = file_path.find("Poe/") + 4
        save_name = "Poe_" + uid + ".nii.gz"
    elif "CENTER-TBI-2020-6" in file_path:
        start_idx = file_path.find("CENTER-TBI-2020-6/") + 18
        save_name = "CENTER-TBI-2020-6_" + uid + ".nii.gz"
    
    return save_name

files_to_force_into_train = [
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/Norman/26337/merged_labels/SWI_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/Norman/26370/merged_labels/SWI_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/Poe/24934/merged_labels/SWI_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/Poe/25927/merged_labels/SWI_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/Poe/26276/merged_labels/SWI_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/CENTER-TBI-2020-6/Sub-028-5cDW456_Site-06-a72b20/merged_labels/FLAIR_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/CENTER-TBI-2020-6/Sub-141-6nBS734_Site-06-a72b20/merged_labels/SWI_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/CENTER-TBI-2020-6/Sub-182-6AOh078_Site-06-a72b20/merged_labels/SWI_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/CENTER-TBI-2020-6/Sub-188-7Zto957_Site-06-a72b20/merged_labels/SWI_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/CENTER-TBI-2020-6/Sub-005-9dbG389_Site-11-0ade7c/merged_labels/SWI_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/CENTER-TBI-2020-6/Sub-007-7TJP797_Site-10-fe5dbb/merged_labels/SWI_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/CENTER-TBI-2020-6/Sub-011-2Ypa486_Site-49-54ceb9/merged_labels/SWI_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/CENTER-TBI-2020-6/Sub-012-9YRr256_Site-49-54ceb9/merged_labels/SWI_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/CENTER-TBI-2020-6/Sub-013-7kvh327_Site-37-8effee/merged_labels/SWI_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/CENTER-TBI-2020-6/Sub-015-6sei477_Site-10-fe5dbb/merged_labels/SWI_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/CENTER-TBI-2020-6/Sub-020-9KsG455_Site-03-9e6a55/merged_labels/SWI_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/CENTER-TBI-2020-6/Sub-030-2avC224_Site-16-ac3478/merged_labels/SWI_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/CENTER-TBI-2020-6/Sub-046-3ReB387_Site-49-54ceb9/merged_labels/SWI_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/CENTER-TBI-2020-6/Sub-058-5Ttp535_Site-49-54ceb9/merged_labels/FLAIR_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/CENTER-TBI-2020-6/Sub-080-8LRN824_Site-16-ac3478/merged_labels/SWI_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/CENTER-TBI-2020-6/Sub-082-4JAm685_Site-16-ac3478/merged_labels/SWI_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/CENTER-TBI-2020-6/Sub-084-9TBb253_Site-16-ac3478/merged_labels/SWI_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/CENTER-TBI-2020-6/Sub-092-6xEr388_Site-49-54ceb9/merged_labels/SWI_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/CENTER-TBI-2020-6/Sub-096-3psT051_Site-49-54ceb9/merged_labels/SWI_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/CENTER-TBI-2020-6/Sub-101-7VXs824_Site-35-9109c8/merged_labels/SWI_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/CENTER-TBI-2020-6/Sub-103-3QBk862_Site-16-ac3478/merged_labels/SWI_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/CENTER-TBI-2020-6/Sub-107-2bAW792_Site-03-9e6a55/merged_labels/SWI_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/CENTER-TBI-2020-6/Sub-108-1DDd610_Site-16-ac3478/merged_labels/SWI_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/CENTER-TBI-2020-6/Sub-118-6HXX623_Site-35-9109c8/merged_labels/SWI_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/CENTER-TBI-2020-6/Sub-119-2zgj625_Site-35-9109c8/merged_labels/FLAIR_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/CENTER-TBI-2020-6/Sub-123-2ByK988_Site-03-9e6a55/merged_labels/SWI_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/CENTER-TBI-2020-6/Sub-163-6CpB238_Site-03-9e6a55/merged_labels/SWI_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/CENTER-TBI-2020-6/Sub-175-5yHv982_Site-03-9e6a55/merged_labels/SWI_mc8June19.nii.gz',
'/mnt/HARDDISK2/data/center-tbi/data/MRI/all_legacy_n_center_1_2020/preprocessed/CENTER-TBI-2020-6/Sub-223-3FRR335_Site-35-9109c8/merged_labels/SWI_mc8June19.nii.gz'
]

def trim_label(label, mask):
    label = label * mask
    return label

def check_if_force_train(file_path):
    id = get_uid(file_path)
    for file in files_to_force_into_train:
        if id in file:
            # print("Forcing file to train: ", id)
            return True
    else:
        return False
def main():
    for i, (FLAIR, SWI, T1, T2, FLAIR_label, SWI_label, merged_label, brain_mask) in enumerate(zip(FLAIR_lines, SWI_lines, T1_lines, T2_lines, labels_FLAIR_lines, labels_SWI_lines, labels_merged_lines, brain_mask_lines)):
        
        try:
            # FLAIR_NIFTI = nib.load(FLAIR)
            # SWI_NIFTI = nib.load(SWI)
            # T1_NIFTI = nib.load(T1)
            # T2_NIFTI = nib.load(T2)
            FLAIR_label_NIFTI = nib.load(FLAIR_label)
            SWI_label_NIFTI = nib.load(SWI_label)
            merged_label_NIFTI = nib.load(merged_label)
            brain_mask_NIFTI = nib.load(brain_mask)
        except Exception as e:
            print(e)

        # print(FLAIR_NIFTI.header)


        # FLAIR_arr = np.array(FLAIR_NIFTI.dataobj)
        # SWI_arr = np.array(SWI_NIFTI.dataobj)
        # T1_arr = np.array(T1_NIFTI.dataobj)
        # T2_arr = np.array(T2_NIFTI.dataobj)
        FLAIR_label_arr = np.array(FLAIR_label_NIFTI.dataobj)
        SWI_label_arr = np.array(SWI_label_NIFTI.dataobj)
        merged_label_arr = np.array(merged_label_NIFTI.dataobj)
        brain_mask_arr = np.array(brain_mask_NIFTI.dataobj)

        # FLAIR_skullstripped = skullstrip_on_mask(FLAIR_arr, brain_mask_arr)
        # SWI_skullstripped = skullstrip_on_mask(SWI_arr, brain_mask_arr)
        # T1_skullstripped = skullstrip_on_mask(T1_arr, brain_mask_arr)
        # T2_skullstripped = skullstrip_on_mask(T2_arr, brain_mask_arr)

        # FLAIR_normed = normalise(FLAIR_skullstripped, brain_mask_arr)
        # SWI_normed = normalise(SWI_skullstripped, brain_mask_arr)
        # T1_normed = normalise(T1_skullstripped, brain_mask_arr)
        # T2_normed = normalise(T2_skullstripped, brain_mask_arr)

        FLAIR_merged_labels = merge_labels(FLAIR_label_arr)
        SWI_merged_labels = merge_labels(SWI_label_arr)
        MERGED_merged_labels = merge_labels(merged_label_arr)

        FLAIR_merged_labels = FLAIR_merged_labels * brain_mask_arr
        SWI_merged_labels = SWI_merged_labels * brain_mask_arr
        MERGED_merged_labels = MERGED_merged_labels * brain_mask_arr

        # stacked_modalities = np.stack((FLAIR_normed, T1_normed, T2_normed, SWI_normed))
        # stacked_modalities = np.transpose(stacked_modalities, (1,2,3,0))

        save_name = create_save_name(FLAIR)
        

        if i % 2 or check_if_force_train(FLAIR):
            # put in train folder
            img_save_path = os.path.join("/home/sedm6251/projectMaterial/datasets/TBI/Train/Images", save_name)
            FLAIR_label_save_path = os.path.join("/home/sedm6251/projectMaterial/datasets/TBI/Train/Labels_FLAIR", save_name)
            SWI_label_save_path = os.path.join("/home/sedm6251/projectMaterial/datasets/TBI/Train/Labels_SWI", save_name)
            merged_label_save_path = os.path.join("/home/sedm6251/projectMaterial/datasets/TBI/Train/Labels_Merged", save_name)    
        else:
            # put in test folder
            img_save_path = os.path.join("/home/sedm6251/projectMaterial/datasets/TBI/Test/Images", save_name)
            FLAIR_label_save_path = os.path.join("/home/sedm6251/projectMaterial/datasets/TBI/Test/Labels_FLAIR", save_name)
            SWI_label_save_path = os.path.join("/home/sedm6251/projectMaterial/datasets/TBI/Test/Labels_SWI", save_name)
            merged_label_save_path = os.path.join("/home/sedm6251/projectMaterial/datasets/TBI/Test/Labels_Merged", save_name)   

        # save_arr_to_nifti(stacked_modalities, img_save_path, None)
        # save_arr_to_nifti(FLAIR_merged_labels, FLAIR_label_save_path, None)
        if save_name == "CENTER-TBI-2020-6_Sub-136-7Qkf235_Site-06-a72b20.nii.gz":
            save_arr_to_nifti(SWI_merged_labels, SWI_label_save_path, None)
            save_arr_to_nifti(FLAIR_merged_labels, FLAIR_label_save_path, None)
            save_arr_to_nifti(MERGED_merged_labels, merged_label_save_path, None)


        # shutil.copy(FLAIR, "/home/sedm6251/projectMaterial/datasets/TBI/orig/Images/FLAIR/" + save_name)
        # shutil.copy(T2, "/home/sedm6251/projectMaterial/datasets/TBI/orig/Images/T2/" + save_name)
        # shutil.copy(T1, "/home/sedm6251/projectMaterial/datasets/TBI/orig/Images/T1/" + save_name)
        # shutil.copy(SWI, "/home/sedm6251/projectMaterial/datasets/TBI/orig/Images/SWI/" + save_name)
        # shutil.copy(FLAIR_label, "/home/sedm6251/projectMaterial/datasets/TBI/orig/Labels_FLAIR/" + save_name)
        # shutil.copy(SWI_label, "/home/sedm6251/projectMaterial/datasets/TBI/orig/Labels_SWI/" + save_name)
        # shutil.copy(merged_label, "/home/sedm6251/projectMaterial/datasets/TBI/orig/Labels_Merged/" + save_name)

            


if __name__ == "__main__":
    main()


