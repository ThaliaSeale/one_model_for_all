import image_normalisation
import numpy as np
import nibabel as nib
import os
if __name__ == "__main__":

    data_path_FLAIR = "/home/sedm6251/projectMaterial/datasets/TBI/Test/Labels_FLAIR/"
    data_path_SWI = "/home/sedm6251/projectMaterial/datasets/TBI/Test/Labels_SWI/"
    data_path_merged = "/home/sedm6251/projectMaterial/datasets/TBI/Test/Labels_Merged/"

    save_dir = "/home/sedm6251/projectMaterial/datasets/TBI/Test/Labels_Multichannel/"

    # labels
	# "0": "background", 
	# "1": "edema",
	# "2": "non-enhancing tumor",
	# "3": "enhancing tumour"
    # merge 2 & 3 as just "pathology" for multi-task purposes and remove edema as a segmentation label

    files_FLAIR = os.listdir(data_path_FLAIR)
    files_SWI = os.listdir(data_path_SWI)
    files_merged = os.listdir(data_path_merged)

    files_FLAIR.sort()
    files_SWI.sort()
    files_merged.sort()

    number_of_files = 156

    for file_name_FLAIR, file_name_SWI, file_name_merged in zip(files_FLAIR, files_SWI, files_merged):

        
        file_path_FLAIR = os.path.join(data_path_FLAIR, file_name_FLAIR)
        file_path_SWI = os.path.join(data_path_SWI, file_name_SWI)
        file_path_merged = os.path.join(data_path_merged, file_name_merged)

        file_FLAIR = nib.load(file_path_FLAIR)
        file_SWI = nib.load(file_path_SWI)
        file_merged = nib.load(file_path_merged)

        FLAIR_arr = np.array(file_FLAIR.dataobj)
        SWI_arr = np.array(file_SWI.dataobj)
        merged_arr = np.array(file_merged.dataobj)

        print("Shapes")
        print(FLAIR_arr.shape)
        print(SWI_arr.shape)
        print(merged_arr.shape)

        two_channel_label = np.stack((FLAIR_arr, SWI_arr, merged_arr),axis=-1)

        nifti_img = nib.Nifti1Image(two_channel_label, None)
        sform = np.diag([1, 1, 1, 1])
        nifti_img.header.set_sform(sform,code="aligned")

        save_path = os.path.join(save_dir, file_name_FLAIR)
        nib.save(nifti_img, save_path)
