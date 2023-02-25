import image_normalisation
import numpy as np
import nibabel as nib
import os
if __name__ == "__main__":

    data_path_inc_edema = "/home/shared_space/data/BRATS_Decathlon_2016_17/BRATS_merged_labels_inc_edema/"
    data_path_exc_edema = "/home/shared_space/data/BRATS_Decathlon_2016_17/BRATS_merged_labels/"
    save_path = "/home/shared_space/data/BRATS_Decathlon_2016_17/two_channel_labels"

    # labels
	# "0": "background", 
	# "1": "edema",
	# "2": "non-enhancing tumor",
	# "3": "enhancing tumour"
    # merge 2 & 3 as just "pathology" for multi-task purposes and remove edema as a segmentation label

    number_of_files = 484

    for i in range(1,number_of_files+1):
        a = str(i).zfill(3)
        print (a + " of " + str(number_of_files))
        
        load_name = "BRATS_" + a + "merged.nii.gz"
        save_name = "BRATS_" + a + "merged.nii.gz"

        nifti_file_inc_edema = image_normalisation.load_nifti(data_path=data_path_inc_edema, file_name=load_name)
        nifti_file_exc_edema = image_normalisation.load_nifti(data_path=data_path_exc_edema, file_name=load_name)



        nifti_array_inc_edema = image_normalisation.nifti_to_array(nifti_file_inc_edema)
        nifti_array_exc_edema = image_normalisation.nifti_to_array(nifti_file_exc_edema)

        print(nifti_array_exc_edema.shape)
        print(nifti_array_inc_edema.shape)

        two_channel_label = np.stack((nifti_array_inc_edema, nifti_array_exc_edema),axis=-1)


        # new_image = nib.Nifti1Image(two_channel_label)
        # save_path = os.path.join(save_path,save_name)
        # nib.save(new_image, save_path)

        image_normalisation.save_arr(two_channel_label, save_path, save_name, nifti_file_inc_edema.affine)
    
    