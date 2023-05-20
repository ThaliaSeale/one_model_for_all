import os
import numpy as np
import nibabel as nib
# from pre_processing.pre_proc_TBI import pre_proc_TBI as preproc
import numpy.ma as ma
import matplotlib.pyplot as plt
from glob import glob


root_dir = "/home/shared_space/data/camcan/camcan_from_ic/camcan_preproc_bg/affine_to_mni"
save_root = "/home/sedm6251/projectMaterial/datasets/camcan2/"

t1_root_dir = os.path.join(root_dir,'T1w/')
t2_root_dir = os.path.join(root_dir,'T2w/')

t1_dir = os.listdir(t1_root_dir)
t1_files = []
brain_mask_files = []
for f in t1_dir:
    if "_unbiased" in f:
        t1_files.append(f)
    if "_brain_mask" in f:
        brain_mask_files.append(f)

t2_dir = os.listdir(t2_root_dir)
t2_files = []
for f in t2_dir:
    if "_unbiased" in f:
        t2_files.append(f)
t1_files.sort()
t2_files.sort()

brain_mask_files.sort()

# print(t1_files[0])
# print(t2_files[0])

for t1_file_name, t2_file_name, brain_mask_name in zip(t1_files, t2_files, brain_mask_files):
    subj_name_t1_ind_start = t1_file_name.find("sub-")
    # subj_name_t2_ind_start = t2_file_path.find("sub-")
    subj_name_t1_ind_end = t1_file_name[subj_name_t1_ind_start:].find("_")
    # subj_name_t2_ind_end = t2_file_path[subj_name_t2_ind_start:].find("T")
    subj_name = t1_file_name[subj_name_t1_ind_start: subj_name_t1_ind_end]
    # subj_name_t2 = t2_file_path[subj_name_t2_ind_start: subj_name_t2_ind_end]

    # # if subj_name_t1 != subj_name_t2:
    # print(subj_name_t1)
    # print(subj_name_t2)

    t1_path = os.path.join(t1_root_dir, t1_file_name)
    t2_path = os.path.join(t2_root_dir, t2_file_name)
    brainmask_path = os.path.join(t1_root_dir, brain_mask_name)

    t1_nifti = nib.load(t1_path)
    t2_nifti = nib.load(t2_path)
    brainmask_nifti = nib.load(brainmask_path)

    t1_arr = np.array(t1_nifti.dataobj)
    t2_arr = np.array(t2_nifti.dataobj)
    brainmask_arr = np.array(brainmask_nifti.dataobj)

    m = brainmask_arr < 0.5

    masked_arr_t1 = ma.masked_array(t1_arr, mask = m)
    mean_t1 = masked_arr_t1.mean()
    std_t1 = masked_arr_t1.std()

    masked_arr_t2 = ma.masked_array(t2_arr, mask = m)
    mean_t2 = masked_arr_t2.mean()
    std_t2 = masked_arr_t2.std()

    normed_t1 = (t1_arr - mean_t1)/std_t1
    normed_t2 = (t2_arr - mean_t2)/std_t2

    normed = np.stack((normed_t1, normed_t2))
    # normed = np.expand_dims(normed, axis=-1)
    normed = np.transpose(normed,[1,2,3,0])

    # plt.imshow(normed[:,:,100])
    # plt.show()

    # print(normed.shape)

    nifti_img = nib.Nifti1Image(normed, None)
    sform = np.diag([1, -1, 1, 1])
    nifti_img.header.set_sform(sform,code="aligned")
    
    save_name = subj_name + ".nii.gz"
    # save_name = "1" + ".nii.gz"
    # print(save_name)
    save_path = os.path.join(save_root,save_name)
    print(save_path)
    nib.save(nifti_img, save_path)


