import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
import numpy.ma as ma
import random

def normalise(arr, mask):
    # print(mask.shape)
    m = ~mask
    masked_arr = ma.masked_array(arr, mask = m)
    mean = masked_arr.mean()
    std = masked_arr.std()

    normed = (arr - mean)/std

    return normed


def create_modality(mod_1: np.array, mod_2: np.array) -> np.array:
    
    coeff_1 = random.randint(2,9)/10
    coeff_2 = 1.0 - coeff_1

    exp_1 = random.randint(5,15)/10
    exp_2 = random.randint(5,15)/10

    mask = (mod_1 > np.min(mod_1))


    # print("coeff 1: ", coeff_1)
    # print("coeff 2: ", coeff_2)
    # print("exp 1: ", exp_1)
    # print("exp 2: ", exp_2)

    exponentiated_1 = np.sign(mod_1) * (np.abs(mod_1) ** exp_1)
    exponentiated_2 = np.sign(mod_2) * (np.abs(mod_2) ** exp_2)
    
    new_modality = (coeff_1 * exponentiated_1) + (coeff_2 * exponentiated_2)
    new_modality = normalise(new_modality, mask)

    # print(np.mean(mod_1))
    # print(np.std(mod_2))

    # print(np.mean(new_modality))
    # print(np.std(new_modality))
    
    return new_modality

    


# if __name__ == "__main__":

#     file = "/home/shared_space/data/BRATS_Decathlon_2016_17/BRATS_Normalised_with_brainmask/normed/BRATS_001_normed_on_mask.nii.gz"
    
#     nifti_file = nib.load(file)
#     image_arr = np.array(nifti_file.dataobj)

#     modalities = [0,1,2,3]
#     num_modalities = len(modalities)

#     number_to_make = 3

#     fig, axs = plt.subplots(number_to_make,3)

#     for i in range(number_to_make):
#         modalities_to_merge = random.sample(modalities,2)

#         mod_1 = image_arr[:,:,:,modalities_to_merge[0]]
#         mod_2 = image_arr[:,:,:,modalities_to_merge[1]]
        
#         new_modality = create_modality(mod_1, mod_2)

#         axs[i,0].imshow(mod_1[:,:,image_arr.shape[2]//2],cmap="gray")
#         axs[i,1].imshow(mod_2[:,:,image_arr.shape[2]//2],cmap="gray")
#         axs[i,2].imshow(new_modality[:,:,image_arr.shape[2]//2],cmap="gray")
#         # axs[i,3].imshow(mask[:,:,image_arr.shape[2]//2],cmap="gray")
    
#     plt.show()



