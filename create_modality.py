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

def show_output(mod_1: np.array, mod_2: np.array, mask: np.array, new_modality: np.array):
    
    fig, axs = plt.subplots(mod_1.shape[0],4)

    for batch_index in range(mod_1.shape[0]):

        axs[batch_index,0].imshow(mod_1[batch_index,0,:,:,mod_1.shape[2]//2],cmap="gray")
        axs[batch_index,1].imshow(mod_2[batch_index,0,:,:,mod_2.shape[2]//2],cmap="gray")
        axs[batch_index,2].imshow(new_modality[batch_index,0,:,:,new_modality.shape[2]//2],cmap="gray")
        axs[batch_index,3].imshow(mask[batch_index,0,:,:,mask.shape[2]//2],cmap="gray")
    
    plt.show()
    print("done")


def create_modality(mod_1: np.array, mod_2: np.array) -> np.array:
    
    coeff_1 = random.randint(2,9)/10
    coeff_2 = 1.0 - coeff_1

    exp_1 = random.randint(5,15)/10
    exp_2 = random.randint(5,15)/10

    # print("coeff 1: ", coeff_1)
    # print("coeff 2: ", coeff_2)
    # print("exp 1: ", exp_1)
    # print("exp 2: ", exp_2)

    exponentiated_1 = np.sign(mod_1) * (np.abs(mod_1) ** exp_1)
    exponentiated_2 = np.sign(mod_2) * (np.abs(mod_2) ** exp_2)
    
    new_modality = (coeff_1 * exponentiated_1) + (coeff_2 * exponentiated_2)


    masks = []
    batch_new_modalities = []
    for batch_index in range(new_modality.shape[0]):
        min_batch_element = np.min(new_modality[batch_index])
        mask = new_modality[batch_index] > min_batch_element

        new_modality_normed = normalise(new_modality[batch_index], mask)
        batch_new_modalities.append(new_modality_normed)
        masks.append(mask)

    new_modality = np.stack(batch_new_modalities)
    masks = np.stack(masks)

    # show_output(mod_1, mod_2, masks, new_modality)
    
    return new_modality

    


if __name__ == "__main__":

    file = "/home/shared_space/data/BRATS_Decathlon_2016_17/BRATS_Normalised_with_brainmask/normed/BRATS_001_normed_on_mask.nii.gz"
    
    nifti_file = nib.load(file)
    image_arr = np.array(nifti_file.dataobj)

    modalities = [0,1,2,3]
    num_modalities = len(modalities)

    number_to_make = 3

    fig, axs = plt.subplots(number_to_make,3)

    for i in range(number_to_make):
        modalities_to_merge = random.sample(modalities,2)

        mod_1 = image_arr[:,:,:,modalities_to_merge[0]]
        mod_2 = image_arr[:,:,:,modalities_to_merge[1]]
        mask = (mod_1 > np.min(mod_1))

        new_modality = create_modality(mod_1, mod_2)

        axs[i,0].imshow(mod_1[:,:,image_arr.shape[2]//2],cmap="gray")
        axs[i,1].imshow(mod_2[:,:,image_arr.shape[2]//2],cmap="gray")
        axs[i,2].imshow(new_modality[:,:,image_arr.shape[2]//2],cmap="gray")
        # axs[i,3].imshow(mask[:,:,image_arr.shape[2]//2],cmap="gray")
    
    plt.show()



