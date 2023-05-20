import os
from monai.data import ImageDataset, decollate_batch, DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import Activations, EnsureChannelFirst, AsDiscrete, Compose, RandRotate90, RandSpatialCrop, ScaleIntensity, CenterSpatialCrop, RandCropByPosNegLabeld, LoadImaged, EnsureChannelFirstd
from monai.utils import set_determinism
import numpy as np
import torch
from itertools import combinations
from HEMIS_Nets_Legacy import hemis_style_net as hemNet
from HEMIS_Nets_Legacy import hemis_style_res_units_after_fusion as hemNet2
from Nets.HEMISv2 import HEMISv2 
from Nets.UNetv2 import UNetv2
from Nets.theory_UNET import theory_UNET
from Nets.multi_scale_fusion_net import MSFN
import nibabel as nib
from glob import glob
import torch
import matplotlib.pyplot as plt

def create_modality_combinations(modalities: list):
  modality_combinations = []
  for i in range(1,len(modalities)+1):
    k = len(modalities)+1-i
    modality_combinations = modality_combinations + list(combinations(modalities,k))
  return modality_combinations
    
def create_dataset(dataset):
  val_imtrans = Compose([EnsureChannelFirst()])
  val_segtrans = Compose([EnsureChannelFirst()])
  if dataset == "BRATS":
    val_size = 40
    img_path_BRATS = "/home/shared_space/data/BRATS_Decathlon_2016_17/BRATS_Normalised_with_brainmask/normed"
    seg_path_BRATS = "/home/shared_space/data/BRATS_Decathlon_2016_17/BRATS_merged_labels_inc_edema"
    # seg_path_BRATS = "/home/shared_space/data/BRATS_Decathlon_2016_17/BRATS_merged_labels"
    images = sorted(glob(os.path.join(img_path_BRATS, "BRATS*_normed_on_mask.nii.gz")))
    segs = sorted(glob(os.path.join(seg_path_BRATS, "BRATS*merged.nii.gz")))
    val_ds = ImageDataset(images[-val_size:], segs[-val_size:], transform=val_imtrans, seg_transform=val_segtrans)
  elif dataset == "ATLAS":
    val_size = 195
    img_path = "/home/sedm6251/projectMaterial/skullStripped/ATLAS/ATLAS/normed_images"
    seg_path = "/home/sedm6251/projectMaterial/skullStripped/ATLAS/ATLAS/trimmed_labels_ints"
    images = sorted(glob(os.path.join(img_path, "*_normed.nii.gz")))
    segs = sorted(glob(os.path.join(seg_path, "*_label_trimmed.nii.gz")))
    val_ds = ImageDataset(images[-val_size:], segs[-val_size:], transform=val_imtrans, seg_transform=val_segtrans)
  elif dataset == "ISLES2015":
    val_size = 8
    img_path = "/home/sedm6251/projectMaterial/baseline_models/ISLES2015/Data/images"
    seg_path = "/home/sedm6251/projectMaterial/baseline_models/ISLES2015/Data/labels"
    images = sorted(glob(os.path.join(img_path, "*_image.nii.gz")))
    segs = sorted(glob(os.path.join(seg_path, "*_label.nii.gz")))
    val_ds = ImageDataset(images[-val_size:], segs[-val_size:], transform=val_imtrans, seg_transform=val_segtrans)
  elif dataset == "MSSEG":
    val_size = 16
    img_path = "/home/sedm6251/projectMaterial/datasets/MSSEG_2016/Normed"
    seg_path = "/home/sedm6251/projectMaterial/datasets/MSSEG_2016/Labels"
    images = sorted(glob(os.path.join(img_path, "*.nii.gz")))
    segs = sorted(glob(os.path.join(seg_path, "*.nii.gz")))
    val_ds = ImageDataset(images[-val_size:], segs[-val_size:], transform=val_imtrans, seg_transform=val_segtrans)
  elif dataset == "TBI":
    val_size = 125
    img_path = "/home/sedm6251/projectMaterial/datasets/TBI/Test/Images"
    #CHANGE THIS DEPENDING ON WHICH SEGS TO USE
    # print("USING TBI NOT MERGED LABELS")
    # seg_path = "/home/sedm6251/projectMaterial/datasets/TBI/Test/Labels_FLAIR"
    # seg_path = "/home/sedm6251/projectMaterial/datasets/TBI/Test/Labels_SWI"
    seg_path = "/home/sedm6251/projectMaterial/datasets/TBI/Test/Labels_Merged"
    images = sorted(glob(os.path.join(img_path, "*.nii.gz")))
    segs = sorted(glob(os.path.join(seg_path, "*.nii.gz")))
    val_ds = ImageDataset(images[-val_size:], segs[-val_size:], transform=val_imtrans, seg_transform=val_segtrans)
  elif dataset == "WMH":
    val_size = 18
    img_path = "/home/sedm6251/projectMaterial/datasets/WMH/Images"
    seg_path = "/home/sedm6251/projectMaterial/datasets/WMH/Segs"
    images = sorted(glob(os.path.join(img_path, "*.nii.gz")))
    segs = sorted(glob(os.path.join(seg_path, "*.nii.gz")))
    val_ds = ImageDataset(images[-val_size:], segs[-val_size:], transform=val_imtrans, seg_transform=val_segtrans)
  elif dataset == "camcan":
    val_size = 250
    img_path = "/home/sedm6251/projectMaterial/datasets/camcan2/"
    # seg_path = "/home/sedm6251/projectMaterial/datasets/camca/Segs"
    images = sorted(glob(os.path.join(img_path, "*.nii.gz")))
    # segs = sorted(glob(os.path.join(seg_path, "*.nii.gz")))
    val_ds = ImageDataset(images[:val_size], images[:val_size], transform=val_imtrans, seg_transform=val_segtrans)

  
  workers=4

  # val_imtrans = Compose([EnsureChannelFirst(strict_check=True, channel_dim="no_channel")])
  

  
  val_loader = DataLoader(val_ds, batch_size=1, num_workers=workers, pin_memory=0)
  return val_loader

def create_net(net, device,cuda_id):
  if net.net_type == "UNet":
    # # single_Unet = monai.networks.nets.UNet(
    # #   spatial_dims=3,
    # #   in_channels=modalities_trained_on,
    # #   out_channels=1,
    # #   kernel_size = (3,3,3),
    # #   channels=(16, 32, 64, 128, 256),
    # #   strides=((2,2,2),(2,2,2),(2,2,2),(2,2,2)),
    # #   num_res_units=res_units,
    # #   dropout=0.2,
    # # ).to(device)
    
    model = theory_UNET(in_channels=net.modalities_trained_on,
                        out_channels=1).to(device)

    model.load_state_dict(torch.load(net.file_path, map_location={"cuda:0":cuda_id,"cuda:1":cuda_id}))
    model.eval()
  if net.net_type == "UNetv2":
    model = UNetv2(
      spatial_dims=3,
      in_channels=net.modalities_trained_on,
      out_channels=1,
      kernel_size = (3,3,3),
      channels=(16, 32, 64, 128, 256),
      strides=((2,2,2),(2,2,2),(2,2,2),(2,2,2)),
      num_res_units=2,
      dropout=0.2,
    ).to(device)
    model.load_state_dict(torch.load(net.file_path, map_location={"cuda:0":cuda_id,"cuda:1":cuda_id}))
    model.eval()
  elif net.net_type == "Theory UNET":
    model = theory_UNET(in_channels=net.modalities_trained_on,
                        out_channels=1).to(device)
    model.load_state_dict(torch.load(net.file_path, map_location={"cuda:0":cuda_id,"cuda:1":cuda_id}))
    model.eval()
  elif net.net_type == "HEMIS_v1":
    # full UNet post fusion
    hemis_net = hemNet.Hemis_Net(device=device)
    hemis_net.load(net.file_path)
    hemis_net.eval()
  elif net.net_type == "HEMIS_v2":
    # 3 Res Units post fusion, 4 outputs per modality UNet
    hemis_net = hemNet2.Hemis_Net(device=device,version=1)
    hemis_net.load(net.file_path)
    hemis_net.eval()
  elif net.net_type == "HEMIS_v3":
    hemis_net = hemNet2.Hemis_Net(device=device, version=2)
    hemis_net.load(net.file_path)
    hemis_net.eval()
  elif net.net_type == "HEM mean":
    hemis_net = HEMISv2(
      post_seg_res_units=False,
      fusion_type="mean",
      UNet_outs=16,
      conv1_in=16,
      conv1_out=16,
      conv2_in=16,
      conv2_out=16,
      conv3_in=16
    ).to(device)
    hemis_net.load_state_dict(torch.load(net.file_path, map_location="cuda:1"))
    hemis_net.eval()
  elif net.net_type == "HEM channel attention":
    hemis_net = HEMISv2(
      post_seg_res_units=False,
      fusion_type="channel attention",
      UNet_outs=16,
      conv1_in=16,
      conv1_out=16,
      conv2_in=16,
      conv2_out=16,
      conv3_in=16
    ).to(device)
    hemis_net.load_state_dict(torch.load(net.file_path, map_location="cuda:1"))
    hemis_net.eval()
  elif net.net_type == "HEM spatial attention":
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
      grid_UNet = True
    ).to(device)
    model.load_state_dict(torch.load(net.file_path, map_location={"cuda:0":cuda_id,"cuda:1":cuda_id}))
    model.eval()
  elif net.net_type == "MSFN":
    model = MSFN(paired=False).to(device)
    model.load_state_dict(torch.load(net.file_path, map_location={"cuda:0":cuda_id,"cuda:1":cuda_id}))
    model.eval()
  elif net.net_type == "MSFNP":
    model = MSFN(paired=True).to(device)
    model.load_state_dict(torch.load(net.file_path, map_location={"cuda:0":cuda_id,"cuda:1":cuda_id}))
    model.eval()
#   elif net_type == "ensemble":
#     msfn_net = MSFN(paired=True).to(device)


  return model

def save_nifti(tensor: torch.Tensor, file_path: str):
    vars_numpy = tensor.cpu().detach().numpy()
    vars_numpy = np.squeeze(vars_numpy)
    # vars_numpy = np.transpose(vars_numpy,(1,2,3,0))
    new_image = nib.Nifti1Image(vars_numpy,affine=None)
    sform = np.diag([1, 1, 1, 1])
    t = [-98,-134,-72,1]
    sform[:,3] = t
    new_image.header.set_sform(sform)
    new_image.header.set_sform(sform,code="aligned")
    nib.save(new_image, file_path)

def plot_slices(tensors: list, slice_numbers, num_columns):
    
    num_rows = len(tensors)//num_columns
    _, axs = plt.subplots(num_rows,num_columns)
    for row in range(num_rows):
      for col in range(num_columns):
        idx = row*num_columns + col
        detached_tesnor = tensors[idx].cpu().detach().numpy()
        axs[row,col].imshow(detached_tesnor[0,0,:,:,slice_numbers[idx]],cmap="gray")

    plt.show()


def create_UNET_input(val_data, modalities, dataset_name, net, modalities_present_at_training):
  zeros_arr = np.zeros_like(val_data[0])
  zeros_arr[:,modalities,:,:,:] = np.array(val_data[0][:,modalities,:,:,:])
  val_data[0] = torch.from_numpy(zeros_arr)

  input_data = torch.from_numpy(np.zeros((1,net.modalities_trained_on,val_data[0].shape[2],val_data[0].shape[3],val_data[0].shape[4]),dtype=np.float32))
  input_data[:,net.channel_map[dataset_name],:,:,:] = val_data[0][:,modalities_present_at_training,:,:,:]
  return input_data


def create_dataloader(val_size: int, images, segs, workers, train_batch_size: int, total_train_data_size: int, current_train_data_size: int, cropped_input_size:list, is_TBI = False):

    div = total_train_data_size//current_train_data_size
    rem = total_train_data_size%current_train_data_size

    train_images = images[:-val_size]
    train_images = train_images * div + train_images[:rem]
    train_segs = segs[:-val_size]
    train_segs = train_segs * div + train_segs[:rem]


    # /////////// TODO REMOVE THIS /////////////////
    # print("USING ONLY FIRST 50 IMAGES!!!!!")
    # train_images = images[:50]
    # train_segs = segs[:50]

    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_segs)]

    train_imtrans = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            # EnsureChannelFirst(strict_check=True),
            # RandSpatialCrop((cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]), random_size=False),
            # RandRotate90(prob=0.1, spatial_axes=(0, 2)),
            RandCropByPosNegLabeld(
              keys=["image", "label"],
              label_key="label",
              spatial_size=(128, 128, 128),
              pos=1,
              neg=1,
              num_samples=1,
              image_key="image",
              image_threshold=0,
            ),
        ]
    )
    train_segtrans = Compose(
        [
            EnsureChannelFirst(strict_check=True),
            RandSpatialCrop((cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]), random_size=False),
            RandRotate90(prob=0.1, spatial_axes=(0, 2)),
        ]
    )
    val_imtrans = Compose([EnsureChannelFirst()])
    val_segtrans = Compose([EnsureChannelFirst()])


    # create a training data loader
    train_ds = Dataset(data=data_dicts, transform=train_imtrans)
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=workers, pin_memory=0)
    # train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=workers)
    # create a validation data loader
    val_ds = ImageDataset(images[-val_size:], segs[-val_size:], transform=val_imtrans, seg_transform=val_segtrans)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=workers, pin_memory=0)

    return train_loader, val_loader