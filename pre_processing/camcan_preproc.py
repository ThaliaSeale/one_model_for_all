import os
import numpy as np
import nibabel as nib
# from pre_processing.pre_proc_TBI import pre_proc_TBI as preproc
import numpy.ma as ma
import matplotlib.pyplot as plt


root_dir = "/home/shared_space/data/camcan/camcan22_preproc/cc700/mri_anat"
save_root = "/home/sedm6251/projectMaterial/datasets/camcan/"

case_folders = os.listdir(root_dir)
case_folders.sort()

for case_folder_name in case_folders:
    
    path_to_case = os.path.join(root_dir, case_folder_name)

    sub_files = os.listdir(path_to_case)
    for f in sub_files:
        if "biascorr" in f:
            t1_file = os.path.join(path_to_case, f)
        if "brainmask" in f :
            brainmask_file = os.path.join(path_to_case, f)

    t1_nifti = nib.load(t1_file)
    brainmask_nifti = nib.load(brainmask_file)

    t1_arr = np.array(t1_nifti.dataobj)
    brainmask_arr = np.array(brainmask_nifti.dataobj)

    m = brainmask_arr < 0.5
    masked_arr = ma.masked_array(t1_arr, mask = m)
    mean = masked_arr.mean()
    std = masked_arr.std()

    normed = (t1_arr - mean)/std
    normed = np.expand_dims(normed, axis=-1)
    normed = np.transpose(normed,[2,0,1,3])

    # plt.imshow(normed[:,:,100])
    # plt.show()

    # print(normed.shape)

    nifti_img = nib.Nifti1Image(normed, None)
    sform = np.diag([1, -1, 1, 1])
    nifti_img.header.set_sform(sform,code="aligned")
    
    save_name = case_folder_name[case_folder_name.find("-")+1:] + ".nii.gz"
    # save_name = "1" + ".nii.gz"
    # print(save_name)
    save_path = os.path.join(save_root,save_name)
    print(save_path)
    nib.save(nifti_img, save_path)


