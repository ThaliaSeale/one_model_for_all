import monai
from monai.data import ImageDataset, create_test_image_3d, decollate_batch, DataLoader, PatchIter, GridPatchDataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import Activations, EnsureChannelFirst, AsDiscrete, Compose, RandRotate90, RandSpatialCrop, ScaleIntensity, CenterSpatialCrop, RandCropByPosNegLabel
from monai.utils import set_determinism
from monai.losses.dice import DiceLoss
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
import os
import sys
from glob import glob
import numpy as np
import random
import sys
from Nets.UNetv2 import UNetv2
from monai.networks.nets.unet import UNet
from Nets.HEMISv2 import HEMISv2
from create_modality import create_modality
from Nets.multi_scale_fusion_net import MSFN
from Nets.theory_UNET import theory_UNET
import utils

# function that creates the dataloader for training
def create_dataloader(val_size: int, images, segs, workers, train_batch_size: int, total_train_data_size: int, current_train_data_size: int, cropped_input_size:list, limited_data = False, limited_data_size = 10):

    div = total_train_data_size//current_train_data_size
    rem = total_train_data_size%current_train_data_size

    train_images = images[:-val_size]
    train_images = train_images * div + train_images[:rem]
    train_segs = segs[:-val_size]
    train_segs = train_segs * div + train_segs[:rem]


    # /////////// TODO ONLY UNCOMMENT FOR LIMITED DATA /////////////////
    if limited_data:
        print("TRAINING ONLY FIRST " + str(limited_data_size) + " IMAGES!!!!!")
        train_images = images[:limited_data_train_size]
        train_segs = segs[:limited_data_train_size]

    # image augmentation through spatial cropping to size and by randomly rotating
    train_imtrans = Compose(
        [
            EnsureChannelFirst(strict_check=True),
            RandSpatialCrop((cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]), random_size=False),
            # RandCropByPosNegLabel((cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]),label=train_segs),
            RandRotate90(prob=0.1, spatial_axes=(0, 2)),
        ]
    )
    val_imtrans = Compose([EnsureChannelFirst()])
    val_segtrans = Compose([EnsureChannelFirst()])

    # create a training data loader
    train_ds = ImageDataset(train_images, train_segs, transform=train_imtrans, seg_transform=train_imtrans)
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=workers, pin_memory=0)
    # create a validation data loader
    val_ds = ImageDataset(images[-val_size:], segs[-val_size:], transform=val_imtrans, seg_transform=val_segtrans)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=workers, pin_memory=0)

    return train_loader, val_loader

# function to randomly set input channels to zero - for use in UNet random dropping
def rand_set_channels_to_zero(dataset_modalities: list, batch_img_data: torch.Tensor):
    number_of_dropped_modalities = np.random.randint(0,len(dataset_modalities))
    modalities_dropped = random.sample(list(np.arange(len(dataset_modalities))), number_of_dropped_modalities)
    modalities_dropped.sort()
    batch_img_data[:,modalities_dropped,:,:,:] = 0.
    modalities_remaining = list(set(np.arange(len(dataset_modalities))) - set(modalities_dropped))
    
    return modalities_remaining, batch_img_data

# function that randomly removes modalities - for use in non-UNet architectures
def remove_random_channels(dataset_modalities: list, batch_img_data: torch.Tensor):
    number_of_dropped_modalities = np.random.randint(0,len(dataset_modalities))
    modalities_dropped = random.sample(list(np.arange(len(dataset_modalities))), number_of_dropped_modalities)
    modalities_remaining = list(set(np.arange(len(dataset_modalities))) - set(modalities_dropped))
    inputs = batch_img_data[:,modalities_remaining,:,:,:]

    return modalities_remaining, inputs

# function for augmentation to add a random modality that is a non-linear combination of two others
def augment(batch_img_data:torch.Tensor):
    modalities_to_merge = random.sample(list(np.arange(batch_img_data.shape[1])),2)
    mod_1 = batch_img_data[:,[modalities_to_merge[0]]]
    mod_2 = batch_img_data[:,[modalities_to_merge[1]]]
    new_modality = create_modality(mod_1, mod_2)
    batch_img_data = torch.cat((batch_img_data, torch.from_numpy(new_modality)), dim=1)

    return batch_img_data
    
# function maps each dataset's modalities to the correct channels of the input
def map_channels(dataset_channels, total_modalities):
        channel_map = []
        for channel in dataset_channels:
            for index, modality in enumerate(total_modalities):
                if channel == modality:
                    channel_map.append(index)
        return channel_map


# main script
if __name__ == "__main__":
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    set_determinism(seed=1)
    print("determinism seed = 1")

    # input arguments
    device_id = int(sys.argv[1])
    epochs = int(sys.argv[2])
    save_name = str(sys.argv[3])
    dataset = str(sys.argv[4])
    randomly_drop = bool(int(sys.argv[5]))
    train_limited_data = bool(int(sys.argv[6]))

    #########################################################
    # *********  ASSUMPTIONS ABOUT THE DATA  **********
    #########################################################
    # THE ORDER OF THE MODALITY CHANNELS IN EACH DATASET IS ASSUMED TO BE:
    # BRATS: ["FLAIR", "T1", "T1c", "T2"]
    # ATLAS: ["T1"]
    # MSSEG: ["FLAIR","T1","T1c","T2", "DP"]
    # ISLES: ["FLAIR", "T1", "T2", "DWI"]
    # TBI: ["FLAIR", "T1", "T2", "SWI"]
    # WMH: ["FLAIR", "T1"]

    # note: DP means PD but should be kept like this for now to avoid affecting any previous alphabetical sorting

    #########################################################
    # *********  START OF CONFIGURABLE PARAMETERS  **********
    #########################################################

    workers = 3
    train_batch_size = 2
    val_interval = 2
    lr = 1e-3

    # settings for refinement training from a pre-trained model
    load_pre_trained_model = True 
    load_model_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG_TBI/UNET/UNET_BRATS_ATLAS_MSSEG_TBI_Epoch_199.pth"
    manual_channel_map = [1,3,5,2] # the slots the modalities of the new dataset should be placed in
    modalities_when_trained = ['DP', 'FLAIR', 'SWI', 'T1', 'T1c', 'T2'] # the modalities the model was trained on (alphabetical order necessary)

    # settings if training on limited data
    # train_limited_data = False
    # limited_data_train_size = 10 #number of training images to use if training on limited data
    if "ISLES" in dataset:
        limited_data_train_size = 10
    if "MSSEG" in dataset:
        limited_data_train_size = 20

    # settings if doing stepwise drop of learning rate
    # drop_learning_rate = False 
    drop_learning_rate = True
    drop_learning_rate_epoch = 500 # epoch at which to decrease the learning rate
    drop_learning_rate_value = 1e-4 # learning rate to drop to

    # set true if using the older unet (net in file UNetv2.py)
    legacy_unet = False

    # parameters that are unlikely to be needed
    is_cluster = False # if training on IBME cluster filepaths
    BRATS_two_channel_seg = False # if using different segmentation ground truths if both T2 and FLAIR are missing then ground truth is without edema
    TBI_multi_channel_seg = True # similar to above but for TBI dataset and SWI missing
    model_type = "UNET" # change this if training a different model
    augment_modalities = False # if True, randomly add a non-linear combination of modalities to the training set for augmentation
    cropped_input_size = [128,128,128] 
    crop_on_label = False # True if random cropping to size should be done centred on areas of lesion

    #########################################################
    # *********  END OF CONFIGURABLE PARAMETERS  **********
    #########################################################

    print("lr: ",lr)
    print("Workers: ", workers)
    print("Batch size: ", train_batch_size)
    print("2 channel seg: ", int(BRATS_two_channel_seg))
    print("Augment modalities?: ", augment_modalities)
    print("RANDOMLY DROP? ",randomly_drop)

    if crop_on_label:
        print("CROPPING ON LABEL")
        img_index = 'image'
        label_index = 'label'
    else:
        print("CROPPING RANDOMLY")
        img_index = 0
        label_index = 1

    # total number of images in the datasets
    total_size_BRATS = 484
    total_size_ATLAS = 654
    total_size_MSSEG = 53
    total_size_ISLES = 28
    total_size_TBI = 281
    total_size_WMH = 60

    # number of images in the training partition - i.e. excluding validation set
    train_size_BRATS = 444
    train_size_ATLAS = 459
    train_size_MSSEG = 37
    train_size_ISLES = 20
    train_size_TBI = 156
    train_size_WMH = 42

    # the set of modalities in each dataset
    channels_BRATS = ["FLAIR", "T1", "T1c", "T2"]
    channels_ATLAS = ["T1"]
    channels_MSSEG = ["FLAIR","T1","T1c","T2","DP"]
    channels_ISLES = ["FLAIR", "T1", "T2", "DWI"]
    channels_TBI = ["FLAIR", "T1", "T2", "SWI"]
    channels_WMH = ["FLAIR", "T1"]

    # calculates the overlapping number of images in a combination of datasets - e.g. if BRATS and ATLAS, it will be the train size of ATLAS (459)
    data_size = max(("BRATS" in dataset) * train_size_BRATS, ("ATLAS" in dataset) * train_size_ATLAS, ("MSSEG" in dataset) * train_size_MSSEG, ("ISLES" in dataset) * train_size_ISLES, ("TBI" in dataset) * train_size_TBI, ("WMH" in dataset) * train_size_WMH)
    
    # finds the union of modalities across datasets
    total_modalities = set(("BRATS" in dataset) * channels_BRATS).union(set(("ATLAS" in dataset) * channels_ATLAS)).union(set(("MSSEG" in dataset) * channels_MSSEG)).union(set(("ISLES" in dataset) * channels_ISLES)).union(set(("TBI" in dataset) * channels_TBI)).union(set(("WMH" in dataset) * channels_WMH))
    total_modalities = sorted(list(total_modalities))

    # generate the mappings of each dataset to the network input channels
    BRATS_channel_map = map_channels(channels_BRATS, total_modalities)
    ATLAS_channel_map = map_channels(channels_ATLAS, total_modalities)
    MSSEG_channel_map = map_channels(channels_MSSEG, total_modalities)
    ISLES_channel_map = map_channels(channels_ISLES, total_modalities)
    TBI_channel_map = map_channels(channels_TBI, total_modalities)
    WMH_channel_map = map_channels(channels_WMH, total_modalities)

    # manually set channel mapping if using a pre-trained model for refinement
    if load_pre_trained_model:
        print("MANUALLY SETTING CHANNEL MAP")
        BRATS_channel_map = manual_channel_map
        ATLAS_channel_map = manual_channel_map
        MSSEG_channel_map = manual_channel_map
        ISLES_channel_map = manual_channel_map
        TBI_channel_map = manual_channel_map
        WMH_channel_map = manual_channel_map

        total_modalities = modalities_when_trained

    
    print("Total modalities: ", total_modalities)
    print("BRATS channel map: ", BRATS_channel_map)
    print("ATLAS channel map: ", ATLAS_channel_map)
    print("MSSEG channel map: ", MSSEG_channel_map)
    print("ISLES channel map: ", ISLES_channel_map)
    print("TBI channel map: ", TBI_channel_map)
    print("WMH channel map: ", WMH_channel_map)

    # arrays to store the dataloaders for each dataset and map them
    train_loaders = []
    val_loaders = []
    data_loader_map = {}

    # BRATS specific setup
    if "BRATS" in dataset:
        print("Training BRATS")
        val_size = total_size_BRATS-train_size_BRATS

        if is_cluster:
            img_path = "/data/sedm6251/BRATS/BRATS_Normalised_with_brainmask/normed"
            if BRATS_two_channel_seg:
                seg_path = "/data/sedm6251/BRATS/two_channel_labels"
            else:
                seg_path = "/data/sedm6251/BRATS/BRATS_merged_labels_inc_edema"
        else:
            img_path = "/home/shared_space/data/BRATS_Decathlon_2016_17/BRATS_Normalised_with_brainmask/normed"
            if BRATS_two_channel_seg:
                seg_path = "/home/shared_space/data/BRATS_Decathlon_2016_17/two_channel_labels"
            else:
                seg_path = "/home/shared_space/data/BRATS_Decathlon_2016_17/BRATS_merged_labels_inc_edema"

        print(seg_path)

        images = sorted(glob(os.path.join(img_path, "BRATS*_normed_on_mask.nii.gz")))
        segs = sorted(glob(os.path.join(seg_path, "BRATS*merged.nii.gz")))

        if crop_on_label:
            train_loader_BRATS, val_loader_BRATS = utils.create_dataloader(val_size=val_size, images=images,segs=segs, workers=workers,train_batch_size=train_batch_size,total_train_data_size=data_size,current_train_data_size=train_size_BRATS,cropped_input_size=cropped_input_size)
        else:   
            train_loader_BRATS, val_loader_BRATS = create_dataloader(val_size=val_size, images=images,segs=segs, workers=workers,train_batch_size=train_batch_size,total_train_data_size=data_size,current_train_data_size=train_size_BRATS,cropped_input_size=cropped_input_size,limited_data=train_limited_data,limited_data_size=limited_data_train_size)
        
        data_loader_map["BRATS"] = len(train_loaders)
        train_loaders.append(train_loader_BRATS)
        val_loaders.append(val_loader_BRATS)

    # ATLAS specific setup
    if "ATLAS" in dataset:
        print("Training ATLAS")
        val_size = total_size_ATLAS-train_size_ATLAS
        if is_cluster:
            img_path_ATLAS = "/data/sedm6251/ATLAS/normed_images"
            seg_path_ATLAS = "/data/sedm6251/ATLAS/trimmed_labels_ints"
        else:
            img_path_ATLAS = "/home/sedm6251/projectMaterial/skullStripped/ATLAS/ATLAS/normed_images"
            seg_path_ATLAS = "/home/sedm6251/projectMaterial/skullStripped/ATLAS/ATLAS/trimmed_labels_ints"
        images = sorted(glob(os.path.join(img_path_ATLAS, "*_normed.nii.gz")))
        segs = sorted(glob(os.path.join(seg_path_ATLAS, "*_label_trimmed.nii.gz")))

        if crop_on_label:
            train_loader_ATLAS, val_loader_ATLAS = utils.create_dataloader(val_size=val_size, images=images,segs=segs, workers=workers,train_batch_size=train_batch_size,total_train_data_size=data_size,current_train_data_size=train_size_ATLAS,cropped_input_size=cropped_input_size)
        else:
            train_loader_ATLAS, val_loader_ATLAS = create_dataloader(val_size=val_size, images=images,segs=segs, workers=workers,train_batch_size=train_batch_size,total_train_data_size=data_size,current_train_data_size=train_size_ATLAS,cropped_input_size=cropped_input_size,limited_data=train_limited_data,limited_data_size=limited_data_train_size)

        data_loader_map["ATLAS"] = len(train_loaders)
        train_loaders.append(train_loader_ATLAS)
        val_loaders.append(val_loader_ATLAS)

    # MSSEG specific setup
    if "MSSEG" in dataset:
        print("Training MSSEG")
        val_size = total_size_MSSEG-train_size_MSSEG

        if is_cluster:
            img_path = "/data/sedm6251/MSSEG/Normed"
            seg_path = "/data/sedm6251/MSSEG/Labels"
        else:
            img_path = "/home/sedm6251/projectMaterial/datasets/MSSEG_2016/Normed"
            seg_path = "/home/sedm6251/projectMaterial/datasets/MSSEG_2016/Labels"
        images = sorted(glob(os.path.join(img_path, "*.nii.gz")))
        segs = sorted(glob(os.path.join(seg_path, "*.nii.gz")))

        if crop_on_label:
            train_loader_MSSEG, val_loader_MSSEG = utils.create_dataloader(val_size=val_size, images=images,segs=segs, workers=workers,train_batch_size=train_batch_size,total_train_data_size=data_size,current_train_data_size=train_size_MSSEG,cropped_input_size=cropped_input_size)
        else:
            train_loader_MSSEG, val_loader_MSSEG = create_dataloader(val_size=val_size, images=images,segs=segs, workers=workers,train_batch_size=train_batch_size,total_train_data_size=data_size,current_train_data_size=train_size_MSSEG,cropped_input_size=cropped_input_size,limited_data=train_limited_data,limited_data_size=limited_data_train_size)

        data_loader_map["MSSEG"] = len(train_loaders)
        train_loaders.append(train_loader_MSSEG)
        val_loaders.append(val_loader_MSSEG)

    # ISLES specific setup
    if "ISLES" in dataset:
        print("Training ISLES")
        val_size = total_size_ISLES-train_size_ISLES

        if is_cluster:
            img_path = "/data/sedm6251/ISLES/images"
            seg_path = "/data/sedm6251/ISLES/labels"
        else:
            img_path = "/home/sedm6251/projectMaterial/baseline_models/ISLES2015/Data/images"
            seg_path = "/home/sedm6251/projectMaterial/baseline_models/ISLES2015/Data/labels"
        images = sorted(glob(os.path.join(img_path, "*.nii.gz")))
        segs = sorted(glob(os.path.join(seg_path, "*.nii.gz")))

        if crop_on_label:
            train_loader_ISLES, val_loader_ISLES = utils.create_dataloader(val_size=val_size, images=images,segs=segs, workers=workers,train_batch_size=train_batch_size,total_train_data_size=data_size,current_train_data_size=train_size_ISLES,cropped_input_size=cropped_input_size)
        else:
            train_loader_ISLES, val_loader_ISLES = create_dataloader(val_size=val_size, images=images,segs=segs, workers=workers,train_batch_size=train_batch_size,total_train_data_size=data_size,current_train_data_size=train_size_ISLES,cropped_input_size=cropped_input_size,limited_data=train_limited_data,limited_data_size=limited_data_train_size)

        data_loader_map["ISLES"] = len(train_loaders)
        train_loaders.append(train_loader_ISLES)
        val_loaders.append(val_loader_ISLES)

    # TBI specific setup
    if "TBI" in dataset:
        print("Training TBI")
        val_size = total_size_TBI-train_size_TBI

        if is_cluster:
            train_img_path = "/data/sedm6251/TBI/Train/Images"
            train_seg_path_FLAIR = "/data/sedm6251/TBI/Train/Labels_FLAIR"
            train_seg_path_SWI = "/data/sedm6251/TBI/Train/Labels_SWI"
            train_seg_path_Merged = "/data/sedm6251/TBI/Train/Labels_Merged"

            val_img_path = "/data/sedm6251/TBI/Test/Images"
            val_seg_path_FLAIR = "/data/sedm6251/TBI/Test/Labels_FLAIR"
            val_seg_path_SWI = "/data/sedm6251/TBI/Test/Labels_SWI"
            val_seg_path_Merged = "/data/sedm6251/TBI/Test/Labels_Merged"

            # ///////////// CHANGE THIS TO CHANGE WHICH SEGMENTATION IS USED //////////////
            
            if TBI_multi_channel_seg:
                train_seg_path = "/data/sedm6251/TBI/Train/Labels_Multichannel"
                val_seg_path = "/data/sedm6251/TBI/Test/Labels_Multichannel"
            else:
                train_seg_path = train_seg_path_Merged
                val_seg_path = val_seg_path_Merged
            print(train_seg_path)
            print(val_seg_path)
        else:
            train_img_path = "/home/sedm6251/projectMaterial/datasets/TBI/Train/Images"
            train_seg_path_FLAIR = "/home/sedm6251/projectMaterial/datasets/TBI/Train/Labels_FLAIR"
            train_seg_path_SWI = "/home/sedm6251/projectMaterial/datasets/TBI/Train/Labels_SWI"
            train_seg_path_Merged = "/home/sedm6251/projectMaterial/datasets/TBI/Train/Labels_Merged"

            val_img_path = "/home/sedm6251/projectMaterial/datasets/TBI/Test/Images"
            val_seg_path_FLAIR = "/home/sedm6251/projectMaterial/datasets/TBI/Test/Labels_FLAIR"
            val_seg_path_SWI = "/home/sedm6251/projectMaterial/datasets/TBI/Test/Labels_SWI"
            val_seg_path_Merged = "/home/sedm6251/projectMaterial/datasets/TBI/Test/Labels_Merged"

            # ///////////// CHANGE THIS TO CHANGE WHICH SEGMENTATION IS USED //////////////
            
            if TBI_multi_channel_seg:
                train_seg_path = "/home/sedm6251/projectMaterial/datasets/TBI/Train/Labels_Multichannel"
                val_seg_path = "/home/sedm6251/projectMaterial/datasets/TBI/Test/Labels_Multichannel"
            else:
                train_seg_path = train_seg_path_Merged
                val_seg_path = val_seg_path_Merged
            print(train_seg_path)
            print(val_seg_path)

        


        images = sorted(glob(os.path.join(train_img_path, "*.nii.gz"))) + sorted(glob(os.path.join(val_img_path, "*.nii.gz")))
        segs = sorted(glob(os.path.join(train_seg_path, "*.nii.gz"))) + sorted(glob(os.path.join(val_seg_path, "*.nii.gz")))

        if crop_on_label:
            train_loader_TBI, val_loader_TBI = utils.create_dataloader(val_size=val_size, images=images,segs=segs, workers=workers,train_batch_size=train_batch_size,total_train_data_size=data_size,current_train_data_size=train_size_TBI,cropped_input_size=cropped_input_size)
        else:
            train_loader_TBI, val_loader_TBI = create_dataloader(val_size=val_size, images=images,segs=segs, workers=workers,train_batch_size=train_batch_size,total_train_data_size=data_size,current_train_data_size=train_size_TBI,cropped_input_size=cropped_input_size,limited_data=train_limited_data,limited_data_size=limited_data_train_size)


        data_loader_map["TBI"] = len(train_loaders)
        train_loaders.append(train_loader_TBI)
        val_loaders.append(val_loader_TBI)

    # WMH speceific setup
    if "WMH" in dataset:
        print("Training WMH")
        val_size = total_size_WMH-train_size_WMH

        if is_cluster:
            img_path = "/data/sedm6251/WMH/Images"
            seg_path = "/data/sedm6251/WMH/Segs"
        else:
            img_path = "/home/sedm6251/projectMaterial/datasets/WMH/Images"
            seg_path = "/home/sedm6251/projectMaterial/datasets/WMH/Segs"

        images = sorted(glob(os.path.join(img_path, "*.nii.gz")))
        segs = sorted(glob(os.path.join(seg_path, "*.nii.gz")))

        if crop_on_label:
            train_loader_WMH, val_loader_WMH = utils.create_dataloader(val_size=val_size, images=images,segs=segs, workers=workers,train_batch_size=train_batch_size,total_train_data_size=data_size,current_train_data_size=train_size_WMH,cropped_input_size=cropped_input_size)
        else:
            train_loader_WMH, val_loader_WMH = create_dataloader(val_size=val_size, images=images,segs=segs, workers=workers,train_batch_size=train_batch_size,total_train_data_size=data_size,current_train_data_size=train_size_WMH,cropped_input_size=cropped_input_size,limited_data=train_limited_data,limited_data_train_size=limited_data_train_size)

        data_loader_map["WMH"] = len(train_loaders)
        train_loaders.append(train_loader_WMH)
        val_loaders.append(val_loader_WMH)


    # ////////////////// ONLY FOR LIMITED DATA  ////////////
    if train_limited_data:
        data_size = limited_data_train_size

    print("Running on GPU:" + str(device_id))
    print("Running for epochs:" + str(epochs))

    cuda_id = "cuda:" + str(device_id)
    device = torch.device(cuda_id)
    torch.cuda.set_device(cuda_id)

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    print("in_channels=",len(total_modalities))
    print("batch size = ",train_batch_size)

    # configuration for different network architectures
    if model_type == "UNET":
        print("TRAINING WITH UNET")

        if not legacy_unet:
            model = theory_UNET(in_channels=len(total_modalities)).to(device)
        else:
            model = UNetv2(
                spatial_dims=3,
                in_channels=len(total_modalities),
                out_channels=1,
                kernel_size = (3,3,3),
                channels=(16, 32, 64, 128, 256),
                strides=((2,2,2),(2,2,2),(2,2,2),(2,2,2)),
                num_res_units=2,
                dropout=0.2,
            ).to(device)

        if load_pre_trained_model:
            print("LOADING MODEL: ", load_model_path)
            model.load_state_dict(torch.load(load_model_path, map_location={"cuda:0":cuda_id,"cuda:1":cuda_id}))
    elif model_type == "HEMIS_spatial_attention":
        print("TRAINING WITH HEMIS SPATIAL ATTENTION")
        model = HEMISv2(
            post_seg_res_units=False,
            fusion_type="spatial attention",
            UNet_outs=12,
            conv1_in=12,
            conv1_out=12,
            conv2_in=12,
            conv2_out=12,
            conv3_in=12,
            pred_uncertainty=False,
            grid_UNet=True,
        ).to(device)
    elif model_type == "MSFN":
        model = MSFN(paired=False).to(device)
    elif model_type == "MSFNP":
        model = MSFN(paired=True).to(device)

    # load pre-trained weights
    if load_pre_trained_model:
            print("LOADING MODEL: ", load_model_path)
            model.load_state_dict(torch.load(load_model_path, map_location={"cuda:0":cuda_id,"cuda:1":cuda_id}))

    # defined loss function and optimiser
    loss_function = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="max", factor=0.2, patience=10, verbose=True,min_lr=lr_lower_lim)
    # cyclic_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=1e-4, max_lr=1e-3, step_size_up=200, mode="triangular",cycle_momentum=False)
    
    # initialise best metric variables
    best_metric_BRATS = -1
    best_metric_ATLAS = -1
    best_metric_MSSEG = -1
    best_metric_ISLES = -1
    best_metric_TBI = -1
    best_metric_WMH = -1

    best_metric_epoch_BRATS = -1
    best_metric_epoch_ATLAS = -1
    best_metric_epoch_MSSEG = -1
    best_metric_epoch_ISLES = -1
    best_metric_epoch_TBI = -1
    best_metric_epoch_WMH = -1

    metric_values = list()

    # create save names for tensorboard logging
    if is_cluster:
        log_save = "/users/sedm6251/tests/runs/" + save_name
        model_save_path = "/users/sedm6251/tests/"
    else:
        log_save = "/home/thalia/one_model_for_all/results/" + save_name + "/log/"
        model_save_path = "/home/thalia/one_model_for_all/results/" + save_name + "/model/" 
        # check directory exists and if not create directory
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
    writer = SummaryWriter(log_dir=log_save)

    # start training cycle
    for epoch in range(epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        # drop learning rate if desired
        if drop_learning_rate and epoch == drop_learning_rate_epoch:
            for g in optimizer.param_groups:
                g['lr'] = drop_learning_rate_value

        # each training loader is zipped together - there are different loaders for each dataset
        for batch_data in zip(*train_loaders):
 
            step += 1

            outputs = []
            labels = []

            # dataset-specific handling of training data - TODO merge all into one function - evolved organically and is messy
            if "BRATS" in dataset:
                loader_index = data_loader_map["BRATS"]
                batch = batch_data[loader_index]

                if model_type == "UNET":

                    if randomly_drop:
                        modalities_remaining, batch[img_index] = rand_set_channels_to_zero(channels_BRATS, batch[img_index])

                        # this part is only relevant for BRATS when doing 2 channel segmentation
                        if (0 not in modalities_remaining) and (3 not in modalities_remaining):
                            # Edema cannot be seen so change segmentation to labels without edema
                            seg_channel = 1
                        else:
                            seg_channel = 0
                        if BRATS_two_channel_seg:
                            label = batch[label_index][:,[seg_channel],:,:,:].to(device)
                        else:
                            label = batch[label_index].to(device)
                    else:
                        label = batch[label_index].to(device)
                    
                    input_data = torch.from_numpy(np.zeros((batch[img_index].shape[0],len(total_modalities),cropped_input_size[0],cropped_input_size[1],cropped_input_size[2]),dtype=np.float32))
                    input_data[:,BRATS_channel_map,:,:,:] = batch[img_index]
                    input_data = input_data.to(device)
                else:
                    if randomly_drop:
                        modalities_remaining, batch[img_index] = remove_random_channels(channels_BRATS, batch[img_index])

                        # this part is only relevant for BRATS when doing 2 channel segmentation
                        if (0 not in modalities_remaining) and (3 not in modalities_remaining):
                            # Edema cannot be seen so change segmentation to labels without edema
                            seg_channel = 1
                        else:
                            seg_channel = 0
                        if BRATS_two_channel_seg:
                            label = batch[label_index][:,[seg_channel],:,:,:].to(device)
                        else:
                            label = batch[label_index].to(device)
                    else:
                        label = batch[label_index].to(device)
                    
                    if augment_modalities:
                        if batch[img_index].shape[label_index] > 1:
                            should_create_modality = bool(random.randint(0,1)) 
                            if should_create_modality:
                                batch[img_index] = augment(batch[img_index])

                    input_data = batch[img_index].to(device)
                
                out = model(input_data)
                outputs.append(out)
                labels.append(label)
                
            if "ATLAS" in dataset:
                loader_index = data_loader_map["ATLAS"]
                batch = batch_data[loader_index]

                if model_type == "UNET":
                    input_data = torch.from_numpy(np.zeros((batch[img_index].shape[0],len(total_modalities),cropped_input_size[0],cropped_input_size[1],cropped_input_size[2]),dtype=np.float32))
                    input_data[:,ATLAS_channel_map,:,:,:] = batch[img_index]
                    input_data = input_data.to(device)
                else:
                    input_data = batch[img_index].to(device)
                label = batch[label_index].to(device)
                out = model(input_data)
                outputs.append(out)
                labels.append(label)

            if "MSSEG" in dataset:
                loader_index = data_loader_map["MSSEG"]
                batch = batch_data[loader_index]

                if model_type == "UNET":
                    if randomly_drop:
                        _, batch[img_index] = rand_set_channels_to_zero(channels_MSSEG, batch[img_index])

                    input_data = torch.from_numpy(np.zeros((batch[img_index].shape[0],len(total_modalities),cropped_input_size[0],cropped_input_size[1],cropped_input_size[2]),dtype=np.float32))
                    input_data[:,MSSEG_channel_map,:,:,:] = batch[img_index]
                    input_data = input_data.to(device)
                else:
                    if randomly_drop:
                        _, batch[img_index] = remove_random_channels(channels_MSSEG, batch[img_index])

                    if augment_modalities:
                        if batch[img_index].shape[1] > 1:
                            should_create_modality = bool(random.randint(0,1))
                            if should_create_modality:
                                batch[img_index] = augment(batch[img_index])
                        

                    input_data = batch[img_index].to(device)
                label = batch[label_index].to(device)
                out = model(input_data)
                outputs.append(out)
                labels.append(label)
                # del input_data
                # del label
                # del out
                # del batch
                # del batch_data
                # torch.cuda.empty_cache()

            if "ISLES" in dataset:
                loader_index = data_loader_map["ISLES"]
                batch = batch_data[loader_index]

                if model_type == "UNET":
                    if randomly_drop:
                        _, batch[img_index] = rand_set_channels_to_zero(channels_ISLES, batch[img_index])                    
                    input_data = torch.from_numpy(np.zeros((batch[img_index].shape[0],len(total_modalities),cropped_input_size[0],cropped_input_size[1],cropped_input_size[2]),dtype=np.float32))
                    input_data[:,ISLES_channel_map,:,:,:] = batch[img_index]
                    input_data = input_data.to(device)
                else:
                    if randomly_drop:
                        _, batch[img_index] = remove_random_channels(channels_ISLES, batch[img_index])
                    if augment_modalities:
                        if batch[img_index].shape[1] > 1:
                            should_create_modality = bool(random.randint(0,1))
                            if should_create_modality:
                                batch[img_index] = augment(batch[img_index])
                    input_data = batch[img_index].to(device)
                label = batch[label_index].to(device)
                out = model(input_data)
                outputs.append(out)
                labels.append(label)

            if "TBI" in dataset:
                loader_index = data_loader_map["TBI"]
                batch = batch_data[loader_index]

                if model_type == "UNET":
                    if randomly_drop:
                        modalities_remaining, batch[img_index] = rand_set_channels_to_zero(channels_TBI, batch[img_index])
                        # this part is only relevant for TBI when doing multi channel segmentation
                        if (0 not in modalities_remaining) and (2 not in modalities_remaining) and (3 not in modalities_remaining):
                            seg_channel = 2
                        elif (0 not in modalities_remaining) and (2 not in modalities_remaining) and (3 in modalities_remaining):
                            seg_channel = 1
                        elif (3 not in modalities_remaining):
                            seg_channel = 0
                        else:
                            seg_channel = 2

                        if TBI_multi_channel_seg:
                            label = batch[label_index][:,[seg_channel],:,:,:].to(device)
                        else:
                            label = batch[label_index].to(device)
                    else:
                        label = batch[label_index].to(device)              
                    
                    input_data = torch.from_numpy(np.zeros((batch[img_index].shape[0],len(total_modalities),cropped_input_size[0],cropped_input_size[1],cropped_input_size[2]),dtype=np.float32))
                    input_data[:,TBI_channel_map,:,:,:] = batch[img_index]
                    input_data = input_data.to(device)

                else:
                    if randomly_drop:
                        modalities_remaining, batch[img_index] = remove_random_channels(channels_TBI, batch[img_index])
                        # this part is only relevant for TBI when doing multi channel segmentation
                        if (0 not in modalities_remaining) and (2 not in modalities_remaining) and (3 not in modalities_remaining):
                            seg_channel = 2
                        elif (0 not in modalities_remaining) and (2 not in modalities_remaining) and (3 in modalities_remaining):
                            seg_channel = 1
                        elif (3 not in modalities_remaining):
                            seg_channel = 0
                        else:
                            seg_channel = 2

                        if TBI_multi_channel_seg:
                            label = batch[label_index][:,[seg_channel],:,:,:].to(device)
                        else:
                            label = batch[label_index].to(device)  
                    else:
                        label = batch[label_index].to(device)
                    if augment_modalities:
                        if batch[img_index].shape[1] > 1:
                            should_create_modality = bool(random.randint(0,1))
                            if should_create_modality:
                                batch[img_index] = augment(batch[img_index])
                    input_data = batch[img_index].to(device)
                
                # label = batch[label_index].to(device)
                out = model(input_data)
                # utils.plot_slices([input_data, input_data, label, label, out, out], [64,100,64,100, 64, 100],2)
                outputs.append(out)
                labels.append(label)

            if "WMH" in dataset:
                loader_index = data_loader_map["WMH"]
                batch = batch_data[loader_index]

                if model_type == "UNET":
                    if randomly_drop:
                        _, batch[img_index] = rand_set_channels_to_zero(channels_WMH, batch[img_index])                    
                    input_data = torch.from_numpy(np.zeros((batch[img_index].shape[0],len(total_modalities),cropped_input_size[0],cropped_input_size[1],cropped_input_size[2]),dtype=np.float32))
                    input_data[:,WMH_channel_map,:,:,:] = batch[img_index]
                    input_data = input_data.to(device)
                else:
                    if randomly_drop:
                        _, batch[img_index] = remove_random_channels(channels_WMH, batch[img_index])
                    if augment_modalities:
                        if batch[img_index].shape[1] > 1:
                            should_create_modality = bool(random.randint(0,1))
                            if should_create_modality:
                                batch[img_index] = augment(batch[img_index])
                    input_data = batch[img_index].to(device)
                label = batch[label_index].to(device)
                out = model(input_data)
                # utils.plot_slices([input_data, input_data, label, label, out, out], [64,100,64,100, 64, 100],2)
                outputs.append(out)
                labels.append(label)


            optimizer.zero_grad()

            combined_outs = torch.cat(outputs, dim=0)
            combined_labels = torch.cat(labels,dim=0)

            loss = loss_function(combined_outs, combined_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = data_size  // train_batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            writer.add_scalar("learning_rate", (optimizer.param_groups)[0]['lr'], epoch_len * epoch + step)
            # cyclic_scheduler.step()
            
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch+1) % 50 == 0:
            
            model_save_name = model_save_path + save_name + "_Epoch_" + str(epoch) + ".pth"
            torch.save(model.state_dict(), model_save_name)
            print("Saved Model")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                seg_channel = 0
                val_images = None
                val_labels = None
                val_outputs = None

                dice_metric.reset()

                if "BRATS" in dataset:

                    loader_index = data_loader_map["BRATS"]
                    for val_data in val_loader_BRATS:
                        if model_type == "UNET":
                            input_data = torch.from_numpy(np.zeros((1,len(total_modalities),val_data[0].shape[2],val_data[0].shape[3],val_data[0].shape[4]),dtype=np.float32))
                            input_data[:,BRATS_channel_map,:,:,:] = val_data[0]
                        else:
                            input_data = val_data[0]
                        input_data = input_data.to(device)

                        if BRATS_two_channel_seg:
                            label = val_data[1][:,[0],:,:,:].to(device)
                        else:
                            label = val_data[1].to(device)
                        
                        roi_size = (cropped_input_size[0], cropped_input_size[1], cropped_input_size[2])
                        sw_batch_size = 1
                        val_outputs = sliding_window_inference(input_data, roi_size, sw_batch_size, model)
                        val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                        # compute metric for current iteration
                        dice_metric(y_pred=val_outputs, y=label)
                    metric_BRATS = dice_metric.aggregate().item()
                    # reset the status for next validation round
                    dice_metric.reset()

                    if metric_BRATS > best_metric_BRATS:
                        best_metric_BRATS = metric_BRATS
                        best_metric_epoch_BRATS = epoch + 1
                        if epoch>1:
                            model_save_name = model_save_path + save_name + "_BEST_BRATS.pth"
                            torch.save(model.state_dict(), model_save_name)
                            print("saved new best metric model for BRATS")
                    print(
                        "current epoch: {} current mean dice BRATS: {:.4f} best mean dice BRATS: {:.4f} at epoch {}".format(
                            epoch + 1, metric_BRATS, best_metric_BRATS, best_metric_epoch_BRATS
                        )
                    )
                    writer.add_scalar("val_mean_dice_BRATS", metric_BRATS, epoch_len * (epoch+1))
                    
                if "ATLAS" in dataset:

                    loader_index = data_loader_map["ATLAS"]
                    for val_data in val_loader_ATLAS:
                        # batch = val_data[loader_index]
                        if model_type == "UNET":
                            input_data = torch.from_numpy(np.zeros((1,len(total_modalities),val_data[0].shape[2],val_data[0].shape[3],val_data[0].shape[4]),dtype=np.float32))
                            input_data[:,ATLAS_channel_map,:,:,:] = val_data[0]
                        else:
                            input_data = val_data[0]
                        input_data = input_data.to(device)
                        label = val_data[1].to(device)
                        
                        roi_size = (cropped_input_size[0], cropped_input_size[1], cropped_input_size[2])
                        sw_batch_size = 1
                        val_outputs = sliding_window_inference(input_data, roi_size, sw_batch_size, model)
                        val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                        # compute metric for current iteration
                        dice_metric(y_pred=val_outputs, y=label)
                    metric_ATLAS = dice_metric.aggregate().item()
                    # reset the status for next validation round
                    dice_metric.reset()

                    if metric_ATLAS > best_metric_ATLAS:
                        best_metric_ATLAS = metric_ATLAS
                        best_metric_epoch_ATLAS = epoch + 1
                        if epoch>1:
                            model_save_name = model_save_path + save_name + "_BEST_ATLAS.pth"
                            torch.save(model.state_dict(), model_save_name)
                            print("saved new best metric model")
                    print(
                        "current epoch: {} current mean dice ATLAS: {:.4f} best mean dice ATLAS: {:.4f} at epoch {}".format(
                            epoch + 1, metric_ATLAS, best_metric_ATLAS, best_metric_epoch_ATLAS
                        )
                    )
                    writer.add_scalar("val_mean_dice_ATLAS", metric_ATLAS, epoch_len * (epoch+1))

                if "MSSEG" in dataset:

                    loader_index = data_loader_map["MSSEG"]
                    for val_data in val_loader_MSSEG:
                        # batch = val_data[loader_index]
                        if model_type == "UNET":
                            input_data = torch.from_numpy(np.zeros((1,len(total_modalities),val_data[0].shape[2],val_data[0].shape[3],val_data[0].shape[4]),dtype=np.float32))
                            input_data[:,MSSEG_channel_map,:,:,:] = val_data[0]
                        else:
                            input_data = val_data[0]
                        input_data = input_data.to(device)
                        label = val_data[1].to(device)
                        
                        roi_size = (cropped_input_size[0], cropped_input_size[1], cropped_input_size[2])
                        sw_batch_size = 1
                        val_outputs = sliding_window_inference(input_data, roi_size, sw_batch_size, model)
                        val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                        # compute metric for current iteration
                        dice_metric(y_pred=val_outputs, y=label)
                    metric_MSSEG = dice_metric.aggregate().item()
                    # reset the status for next validation round
                    dice_metric.reset()

                    if metric_MSSEG > best_metric_MSSEG:
                        best_metric_MSSEG = metric_MSSEG
                        best_metric_epoch_MSSEG = epoch + 1
                        if epoch>1:
                            model_save_name = model_save_path + save_name + "_BEST_MSSEG.pth"
                            torch.save(model.state_dict(), model_save_name)
                            print("saved new best metric model")
                    print(
                        "current epoch: {} current mean dice MSSEG: {:.4f} best mean dice MSSEG: {:.4f} at epoch {}".format(
                            epoch + 1, metric_MSSEG, best_metric_MSSEG, best_metric_epoch_MSSEG
                        )
                    )
                    writer.add_scalar("val_mean_dice_MSSEG", metric_MSSEG, epoch_len * (epoch+1))

                if "ISLES" in dataset:

                    loader_index = data_loader_map["ISLES"]
                    for val_data in val_loader_ISLES:
                        # batch = val_data[loader_index]
                        if model_type == "UNET":
                            input_data = torch.from_numpy(np.zeros((1,len(total_modalities),val_data[0].shape[2],val_data[0].shape[3],val_data[0].shape[4]),dtype=np.float32))
                            input_data[:,ISLES_channel_map,:,:,:] = val_data[0]
                        else:
                            input_data = val_data[0]
                        input_data = input_data.to(device)
                        label = val_data[1].to(device)
                        
                        roi_size = (cropped_input_size[0], cropped_input_size[1], cropped_input_size[2])
                        sw_batch_size = 1
                        val_outputs = sliding_window_inference(input_data, roi_size, sw_batch_size, model)
                        val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                        # compute metric for current iteration
                        dice_metric(y_pred=val_outputs, y=label)
                    metric_ISLES = dice_metric.aggregate().item()
                    # reset the status for next validation round
                    dice_metric.reset()

                    if metric_ISLES > best_metric_ISLES:
                        best_metric_ISLES = metric_ISLES
                        best_metric_epoch_ISLES = epoch + 1
                        if epoch>1:
                            model_save_name = model_save_path + save_name + "_BEST_ISLES.pth"
                            torch.save(model.state_dict(), model_save_name)
                            print("saved new best metric model")
                    print(
                        "current epoch: {} current mean dice ISLES: {:.4f} best mean dice ISLES: {:.4f} at epoch {}".format(
                            epoch + 1, metric_ISLES, best_metric_ISLES, best_metric_epoch_ISLES
                        )
                    )
                    writer.add_scalar("val_mean_dice_ISLES", metric_ISLES, epoch_len * (epoch+1))

                if "TBI" in dataset:

                    loader_index = data_loader_map["TBI"]
                    for val_data in val_loader_TBI:
                        # batch = val_data[loader_index]
                        if model_type == "UNET":
                            input_data = torch.from_numpy(np.zeros((1,len(total_modalities),val_data[0].shape[2],val_data[0].shape[3],val_data[0].shape[4]),dtype=np.float32))
                            input_data[:,TBI_channel_map,:,:,:] = val_data[0]
                        else:
                            input_data = val_data[0]
                        input_data = input_data.to(device)

                        if TBI_multi_channel_seg:
                            label = val_data[1][:,[2],:,:,:].to(device)
                        else:
                            label = val_data[1].to(device)
                        # label = val_data[1].to(device)
                        
                        roi_size = (cropped_input_size[0], cropped_input_size[1], cropped_input_size[2])
                        sw_batch_size = 1
                        val_outputs = sliding_window_inference(input_data, roi_size, sw_batch_size, model)
                        val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                        # compute metric for current iteration
                        dice_metric(y_pred=val_outputs, y=label)
                    metric_TBI = dice_metric.aggregate().item()
                    # reset the status for next validation round
                    dice_metric.reset()

                    if metric_TBI > best_metric_TBI:
                        best_metric_TBI = metric_TBI
                        best_metric_epoch_TBI = epoch + 1
                        if epoch>1:
                            model_save_name = model_save_path + save_name + "_BEST_TBI.pth"
                            torch.save(model.state_dict(), model_save_name)
                            print("saved new best metric model")
                    print(
                        "current epoch: {} current mean dice TBI: {:.4f} best mean dice TBI: {:.4f} at epoch {}".format(
                            epoch + 1, metric_TBI, best_metric_TBI, best_metric_epoch_TBI
                        )
                    )
                    writer.add_scalar("val_mean_dice_TBI", metric_TBI, epoch_len * (epoch+1))

                if "WMH" in dataset:

                    loader_index = data_loader_map["WMH"]
                    for val_data in val_loader_WMH:
                        # batch = val_data[loader_index]
                        if model_type == "UNET":
                            input_data = torch.from_numpy(np.zeros((1,len(total_modalities),val_data[0].shape[2],val_data[0].shape[3],val_data[0].shape[4]),dtype=np.float32))
                            input_data[:,WMH_channel_map,:,:,:] = val_data[0]
                        else:
                            input_data = val_data[0]
                        input_data = input_data.to(device)
                        label = val_data[1].to(device)
                        
                        roi_size = (cropped_input_size[0], cropped_input_size[1], cropped_input_size[2])
                        sw_batch_size = 1
                        val_outputs = sliding_window_inference(input_data, roi_size, sw_batch_size, model)
                        val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                        # compute metric for current iteration
                        dice_metric(y_pred=val_outputs, y=label)
                    metric_WMH = dice_metric.aggregate().item()
                    # reset the status for next validation round
                    dice_metric.reset()

                    if metric_WMH > best_metric_WMH:
                        best_metric_WMH = metric_WMH
                        best_metric_epoch_WMH = epoch + 1
                        if epoch>1:
                            model_save_name = model_save_path + save_name + "_BEST_WMH.pth"
                            torch.save(model.state_dict(), model_save_name)
                            print("saved new best metric model")
                    print(
                        "current epoch: {} current mean dice WMH: {:.4f} best mean dice WMH: {:.4f} at epoch {}".format(
                            epoch + 1, metric_WMH, best_metric_WMH, best_metric_epoch_WMH
                        )
                    )
                    writer.add_scalar("val_mean_dice_WMH", metric_WMH, epoch_len * (epoch+1))

                # scheduler.step(metric,epoch=epoch)
                # writer.add_scalar("learning_rate", (optimizer.param_groups)[0]['lr'], epoch_len * (epoch+1))


    # print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()

