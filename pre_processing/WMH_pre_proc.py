import os
import numpy as np
import nibabel as nib
import pre_proc_TBI.pre_proc_TBI as preproc

def main():
    root_folder = "/home/shared_space/data/wmh-2017/preprocessed_dm_all"
    patients = os.listdir(root_folder)
    patients.sort()

    for patient in patients:
        patient_dir = os.path.join(root_folder, patient)

        FLAIR_file_path = os.path.join(patient_dir, "channel_FLAIR/FLAIR.nii.gz")
        T1_file_path = os.path.join(patient_dir, "channel_T1/T1.nii.gz")
        ROI_mask_file_path = os.path.join(patient_dir, "roi_mask/Brain_Mask.nii.gz")
        GT_file_path = os.path.join(patient_dir, "ground_truth/wmh.nii.gz")

        FLAIR_file = nib.load(FLAIR_file_path)
        T1_file = nib.load(T1_file_path)
        ROI_mask_file = nib.load(ROI_mask_file_path)
        GT_file = nib.load(GT_file_path)

        FLAIR_arr = np.array(FLAIR_file.dataobj)
        T1_arr = np.array(T1_file.dataobj)
        ROI_mask_arr = np.array(ROI_mask_file.dataobj)
        GT_arr = np.array(GT_file.dataobj)

        FLAIR_skull_strip = preproc.skullstrip_on_mask(FLAIR_arr, ROI_mask_arr)
        T1_skull_strip = preproc.skullstrip_on_mask(T1_arr, ROI_mask_arr)

        FLAIR_normed = preproc.normalise(FLAIR_skull_strip, ROI_mask_arr)
        T1_normed = preproc.normalise(T1_skull_strip, ROI_mask_arr)

        GT_merged = preproc.merge_labels(GT_arr)
        GT_trimmed = preproc.trim_label(GT_merged, ROI_mask_arr)

        stacked_modalities = np.stack((FLAIR_normed, T1_normed))
        stacked_modalities = np.transpose(stacked_modalities, (1,2,3,0))

        save_name = patient + ".nii.gz"

        img_save_path = os.path.join("/home/sedm6251/projectMaterial/datasets/WMH/Images", save_name)
        seg_save_path = os.path.join("/home/sedm6251/projectMaterial/datasets/WMH/Segs", save_name)

        preproc.save_arr_to_nifti(stacked_modalities, img_save_path, None)
        preproc.save_arr_to_nifti(GT_trimmed, seg_save_path, None)









if __name__ == "__main__":
    main()