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

    pretrained_model_path = "results/23_06__14_26_exc_WMH/models/23_06__14_26_exc_WMH23_06__14_26_exc_WMH_Epoch_549.pth" 
    pretrained_model = theory_UNET(in_channels = 6,
                                   out_channels=1)
    manual_channel_map = [0,1]
    modalities_when_trained = ["FLAIR", "T1"]

    # settings if doing stepwise drop of learning rate
    drop_learning_rate = True
    drop_learning_rate_epoch = 150 # epoch at which to decrease the learning rate
    drop_learning_rate_value = 1e-4 # learning rate to drop to

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

    # channel mappings 
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
            train_loader_BRATS, val_loader_BRATS = create_dataloader(val_size=val_size, images=images,segs=segs, workers=workers,train_batch_size=train_batch_size,total_train_data_size=data_size,current_train_data_size=train_size_BRATS,cropped_input_size=cropped_input_size)
        
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
            train_loader_ATLAS, val_loader_ATLAS = create_dataloader(val_size=val_size, images=images,segs=segs, workers=workers,train_batch_size=train_batch_size,total_train_data_size=data_size,current_train_data_size=train_size_ATLAS,cropped_input_size=cropped_input_size)

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
            train_loader_MSSEG, val_loader_MSSEG = create_dataloader(val_size=val_size, images=images,segs=segs, workers=workers,train_batch_size=train_batch_size,total_train_data_size=data_size,current_train_data_size=train_size_MSSEG,cropped_input_size=cropped_input_size)

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
            train_loader_ISLES, val_loader_ISLES = create_dataloader(val_size=val_size, images=images,segs=segs, workers=workers,train_batch_size=train_batch_size,total_train_data_size=data_size,current_train_data_size=train_size_ISLES,cropped_input_size=cropped_input_size)

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
            train_loader_TBI, val_loader_TBI = create_dataloader(val_size=val_size, images=images,segs=segs, workers=workers,train_batch_size=train_batch_size,total_train_data_size=data_size,current_train_data_size=train_size_TBI,cropped_input_size=cropped_input_size)


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
            train_loader_WMH, val_loader_WMH = create_dataloader(val_size=val_size, images=images,segs=segs, workers=workers,train_batch_size=train_batch_size,total_train_data_size=data_size,current_train_data_size=train_size_WMH,cropped_input_size=cropped_input_size)

        data_loader_map["WMH"] = len(train_loaders)
        train_loaders.append(train_loader_WMH)
        val_loaders.append(val_loader_WMH)

    cuda_id = "cuda:" + str(device_id)
    device = torch.device(cuda_id)
    torch.cuda.set_device(cuda_id)

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    print("in_channels=",len(total_modalities))
    print("batch size = ",train_batch_size)

    # configure pretrained model
    print("LOADING PRETRAINED MODEL:", pretrained_model_path)
    pretrained_model.load_state_dict(torch.load(pretrained_model_path, map_location={"cuda:0":cuda_id,"cuda:1":cuda_id}))