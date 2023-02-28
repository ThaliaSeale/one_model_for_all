import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import monai
from monai.data import ImageDataset, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import Activations, EnsureChannelFirst, AsDiscrete, Compose, RandRotate90, RandSpatialCrop, ScaleIntensity, CenterSpatialCrop
from monai.utils import set_determinism
import numpy as np
import torch
from torchmetrics.classification import Recall, JaccardIndex
import random
from itertools import combinations
import sys
from HEMIS_Nets_Legacy import hemis_style_net as hemNet
from HEMIS_Nets_Legacy import hemis_style_res_units_after_fusion as hemNet2
from HEMISv2 import HEMISv2 
from Nets.UNetv2 import UNetv2
from Nets.multi_scale_fusion_net import MSFN
import nibabel as nib
import epistemic_unc


from glob import glob



def create_modality_combinations(modalities: list):
  modality_combinations = []
  for i in range(1,len(modalities)+1):
    k = len(modalities)+1-i
    modality_combinations = modality_combinations + list(combinations(modalities,k))
  return modality_combinations
    
def create_dataset(dataset):
  if dataset == "BRATS":
    val_size = 40
    img_path_BRATS = "/home/shared_space/data/BRATS_Decathlon_2016_17/BRATS_Normalised_with_brainmask/normed"
    seg_path_BRATS = "/home/shared_space/data/BRATS_Decathlon_2016_17/BRATS_merged_labels_inc_edema"
    images = sorted(glob(os.path.join(img_path_BRATS, "BRATS*_normed_on_mask.nii.gz")))
    segs = sorted(glob(os.path.join(seg_path_BRATS, "BRATS*merged.nii.gz")))
  elif dataset == "ATLAS":
    val_size = 195
    img_path = "/home/sedm6251/projectMaterial/skullStripped/ATLAS/ATLAS/normed_images"
    seg_path = "/home/sedm6251/projectMaterial/skullStripped/ATLAS/ATLAS/trimmed_labels_ints"
    images = sorted(glob(os.path.join(img_path, "*_normed.nii.gz")))
    segs = sorted(glob(os.path.join(seg_path, "*_label_trimmed.nii.gz")))
  elif dataset == "ISLES2015":
    val_size = 28
    img_path = "/home/sedm6251/projectMaterial/baseline_models/ISLES2015/Data/images"
    seg_path = "/home/sedm6251/projectMaterial/baseline_models/ISLES2015/Data/labels"
    images = sorted(glob(os.path.join(img_path, "*_image.nii.gz")))
    segs = sorted(glob(os.path.join(seg_path, "*_label.nii.gz")))
  elif dataset == "MSSEG":
    val_size = 16
    img_path = "/home/sedm6251/projectMaterial/datasets/MSSEG_2016/Normed"
    seg_path = "/home/sedm6251/projectMaterial/datasets/MSSEG_2016/Labels"
    images = sorted(glob(os.path.join(img_path, "*.nii.gz")))
    segs = sorted(glob(os.path.join(seg_path, "*.nii.gz")))

  
  workers=4

  # val_imtrans = Compose([EnsureChannelFirst(strict_check=True, channel_dim="no_channel")])
  val_imtrans = Compose([EnsureChannelFirst()])
  val_segtrans = Compose([EnsureChannelFirst()])

  val_ds = ImageDataset(images[-val_size:], segs[-val_size:], transform=val_imtrans, seg_transform=val_segtrans)
  val_loader = DataLoader(val_ds, batch_size=1, num_workers=workers, pin_memory=0)
  return val_loader

def create_net(net_path, net_type, device, modalities_trained_on,cuda_id):
  if net_type == "UNet":
    # single_Unet = monai.networks.nets.UNet(
    #   spatial_dims=3,
    #   in_channels=modalities_trained_on,
    #   out_channels=1,
    #   kernel_size = (3,3,3),
    #   channels=(16, 32, 64, 128, 256),
    #   strides=((2,2,2),(2,2,2),(2,2,2),(2,2,2)),
    #   num_res_units=res_units,
    #   dropout=0.2,
    # ).to(device)
    model = UNetv2(
      spatial_dims=3,
      in_channels=modalities_trained_on,
      out_channels=1,
      kernel_size = (3,3,3),
      channels=(16, 32, 64, 128, 256),
      strides=((2,2,2),(2,2,2),(2,2,2),(2,2,2)),
      num_res_units=2,
      dropout=0.2,
    ).to(device)
    model.load_state_dict(torch.load(net_path, map_location={"cuda:0":cuda_id,"cuda:1":cuda_id}))
    model.eval()
  elif net_type == "HEMIS_v1":
    # full UNet post fusion
    hemis_net = hemNet.Hemis_Net(device=device)
    hemis_net.load(net_path)
    hemis_net.eval()
  elif net_type == "HEMIS_v2":
    # 3 Res Units post fusion, 4 outputs per modality UNet
    hemis_net = hemNet2.Hemis_Net(device=device,version=1)
    hemis_net.load(net_path)
    hemis_net.eval()
  elif net_type == "HEMIS_v3":
    hemis_net = hemNet2.Hemis_Net(device=device, version=2)
    hemis_net.load(net_path)
    hemis_net.eval()
  elif net_type == "HEM mean":
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
    hemis_net.load_state_dict(torch.load(net_path, map_location="cuda:1"))
    hemis_net.eval()
  elif net_type == "HEM channel attention":
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
    hemis_net.load_state_dict(torch.load(net_path, map_location="cuda:1"))
    hemis_net.eval()
  elif net_type == "HEM spatial attention":
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
    model.load_state_dict(torch.load(net_path, map_location={"cuda:0":cuda_id,"cuda:1":cuda_id}))
    model.eval()
  elif net_type == "MSFN":
    model = MSFN(paired=False).to(device)
    model.load_state_dict(torch.load(net_path, map_location={"cuda:0":cuda_id,"cuda:1":cuda_id}))
    model.eval()
  elif net_type == "MSFNP":
    model = MSFN(paired=True).to(device)
    model.load_state_dict(torch.load(net_path, map_location={"cuda:0":cuda_id,"cuda:1":cuda_id}))
    model.eval()

  return model

def test(model, val_loader, modalities, device, channel_map, modalities_present_at_training, net_type,modalities_trained_on, save_outputs, save_path="/home/sedm2651"):
  cropped_input_size = [128,128,128]
  dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
  post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

  with torch.no_grad():
    recall = Recall('binary').to(device)
    recall_total = 0.

    jaccard = JaccardIndex('binary').to(device)
    jaccard_total = 0.
    val_images = None
    val_labels = None
    val_outputs = None
    steps = 0
    cumulative_dropped = 0

    dice_metric.reset()
    for val_data in val_loader:

      if net_type == "UNet":

        zeros_arr = np.zeros_like(val_data[0])
        zeros_arr[:,modalities,:,:,:] = np.array(val_data[0][:,modalities,:,:,:])
        val_data[0] = torch.from_numpy(zeros_arr)

        input_data = torch.from_numpy(np.zeros((1,modalities_trained_on,val_data[0].shape[2],val_data[0].shape[3],val_data[0].shape[4]),dtype=np.float32))
        input_data[:,channel_map,:,:,:] = val_data[0][:,modalities_present_at_training,:,:,:]
        val_data[0] = input_data
        
      else:
        val_data[0] = val_data[0][:,modalities,:,:,:]
      val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
      roi_size = (cropped_input_size[0], cropped_input_size[1], cropped_input_size[2])
      sw_batch_size = 1

      if net_type == "UNet":
        ep = epistemic_unc.Epistemic()
        mean_out, var_mask = ep.calculate_unc(model, val_images)
        val_outputs = mean_out
        # val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
      elif net_type in ("HEMIS_v1", "HEMIS_v2","HEMIS_v3"):
        # val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, hemis_net.combined_model)
        pass
      elif net_type == "HEM mean" or net_type == "HEM channel attention" or net_type == "HEM spatial attention":
        val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
      else:
        ep = epistemic_unc.Epistemic()
        mean_out, var_mask, var = ep.calculate_unc(model, val_images)
        val_outputs = mean_out
        # val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
      
      val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]


      vars_numpy = var.cpu().detach().numpy()
      vars_numpy = np.squeeze(vars_numpy)
      # vars_numpy = np.transpose(vars_numpy,(1,2,3,0))
      new_image = nib.Nifti1Image(vars_numpy,affine=None)
      sform = np.diag([1, 1, 1, 1])
      # t = [-98,-134,-72,1]
      # sform[:,3] = t
      new_image.header.set_sform(sform)
      nib.save(new_image,"/home/sedm6251/projectMaterial/baseline_models/Combined_Training/Test_ISLES/MSFN_BRATS_ATLAS_MSSEG_BEST_BRATS_FLAIR_T1_T2_DWI/EP_UNC/ep_var_"+str(steps) + ".nii.gz")

      output_numpy = val_outputs[0].cpu().detach().numpy()
      output_numpy = np.squeeze(output_numpy)
      # output_numpy = np.transpose(output_numpy,(1,2,3,0))
      new_image = nib.Nifti1Image(output_numpy,affine=None)
      sform = np.diag([1, 1, 1, 1])
      # t = [-98,-134,-72,1]
      # sform[:,3] = t
      new_image.header.set_sform(sform, code='aligned')
      nib.save(new_image,"/home/sedm6251/projectMaterial/baseline_models/Combined_Training/Test_ISLES/MSFN_BRATS_ATLAS_MSSEG_BEST_BRATS_FLAIR_T1_T2_DWI/EP_OUT/ep_out_"+str(steps) + ".nii.gz")



      val_outputs = torch.mul(val_outputs[0],var_mask)

      # manual DWI for MSFN BRATS ATLAS MSSEG - ISLES val data
      # if modalities == (0,1,2,3):
      #   if steps==0:
      #     val_outputs[0][:,:,:,0:46] = 0.
      #   elif steps == 1:
      #     val_outputs[0][:,:,:,0:51] = 0.
      #   elif steps == 6 or steps == 7 or steps==4:
      #     val_outputs[0][:,:,:,0:64] = 0.

      # manual DWI for HEMIS BRATS ATLAS MSSEG - ISLES val data
      # if modalities == (0,1,2,3):
      #   if steps == 1:
      #     val_outputs[0][:,:,:,0:51] = 0.
      #   elif steps == 6 or steps == 7:
      #     val_outputs[0][:,:,:,0:64] = 0.

      #manual WI for MSFN BRATS ATLAS MSSEG - ISLES all images
      # if modalities == (0,1,2,3):
      #   if steps==0:
      #     val_outputs[0][:,:,:,0:59] = 0.
      #   elif steps == 1:
      #     val_outputs[0][:,:,:,0:56] = 0.
      #   elif steps == 2:
      #     val_outputs[0][:,:,:,0:38] = 0.
      #   elif steps == 4:
      #     val_outputs[0][:,:,:,0:48] = 0.
      #   elif steps == 6:
      #     val_outputs[0][:,:,:,0:56] = 0.
      #   elif steps == 7:
      #     val_outputs[0][:,:,:,0:61] = 0.
      #   elif steps == 8:
      #     val_outputs[0][:,:,:,0:54] = 0.
      #   elif steps == 9:
      #     val_outputs[0][:,:,:,0:49] = 0.
      #   elif steps == 10:
      #     val_outputs[0][:,:,:,0:49] = 0.
      #   elif steps == 13:
      #     val_outputs[0][:,:,:,0:66] = 0.
      #   elif steps == 14:
      #     val_outputs[0][:,:,:,46:56] = 0.
      #   elif steps == 16:
      #     val_outputs[0][:,:,:,0:66] = 0.
      #   elif steps == 17:
      #     val_outputs[0][:,:,:,0:56] = 0.
      #   elif steps == 18:
      #     val_outputs[0][:,:,:,0:52] = 0.
      #   elif steps == 20:
      #     val_outputs[0][:,:,:,0:47] = 0.
      #   elif steps == 21:
      #     val_outputs[0][:,:,:,0:51] = 0.
      #   elif steps == 24:
      #     val_outputs[0][:,:,:,0:65] = 0.
      #   elif steps == 26:
      #     val_outputs[0][:,:,:,0:66] = 0.
      #   elif steps == 27:
      #     val_outputs[0][:,:,:,0:66] = 0.

      #manual DWI for MSFN PAIRED BRATS ATLAS MSSEG - ISLES all images
      # if modalities == (0,1,2,3):
      #   print("manually trimming DWI")
      #   if steps==0:
      #     val_outputs[0][:,:,:,0:59] = 0.
      #   elif steps == 1:
      #     val_outputs[0][:,:,:,0:56] = 0.
      #   elif steps == 2:
      #     val_outputs[0][:,:,:,0:38] = 0.
      #   elif steps ==3:
      #     val_outputs[0][:,:,:,0:53] = 0.
      #   elif steps == 4:
      #     val_outputs[0][:,:,:,0:60] = 0.
      #   elif steps == 6:
      #     val_outputs[0][:,:,:,0:57] = 0.
      #   elif steps == 7:
      #     val_outputs[0][:,:,:,0:66] = 0.
      #   elif steps == 8:
      #     val_outputs[0][:,:,:,0:54] = 0.
      #   elif steps == 9:
      #     val_outputs[0][:,:,:,0:49] = 0.
      #   elif steps == 10:
      #     val_outputs[0][:,:,:,0:49] = 0.
      #   elif steps ==11:
      #     val_outputs[0][:,:,:,0:66] = 0.
      #   elif steps == 13:
      #     val_outputs[0][:,:,:,0:71] = 0.
      #   elif steps == 14:
      #     val_outputs[0][:,:,:,41:61] = 0.
      #   elif steps == 16:
      #     val_outputs[0][:,:,:,0:67] = 0.
      #   elif steps == 17:
      #     val_outputs[0][:,:,:,0:56] = 0.
      #   elif steps == 18:
      #     val_outputs[0][:,:,:,0:71] = 0.
      #   elif steps == 20:
      #     val_outputs[0][:,:,:,0:47] = 0.
      #   elif steps == 21:
      #     val_outputs[0][:,:,:,0:51] = 0.
      #   elif steps == 23:
      #     val_outputs[0][:,:,:,0:56] = 0.
      #   elif steps == 24:
      #     val_outputs[0][:,:,:,0:66] = 0.
      #   elif steps == 26:
      #     val_outputs[0][:,:,:,0:66] = 0.
      #   elif steps == 27:
      #     val_outputs[0][:,:,:,0:66] = 0.


      if save_outputs:
        outputs_numpy = val_outputs[0].cpu().detach().numpy()
        outputs_numpy = np.transpose(outputs_numpy,(1,2,3,0))
        new_image = nib.Nifti1Image(outputs_numpy,affine=None)
        new_image.header.set_sform(np.diag([1, 1, 1, 1]), code='aligned')
        file_save_path = save_path + str(steps) + ".nii.gz"
        nib.save(new_image, file_save_path)


      # compute metric for current iteration
      dice_metric(y_pred=val_outputs, y=val_labels)
      recall_total += float(recall(val_outputs[0],val_labels[0]).cpu().detach())
      jaccard_total+= float(jaccard(val_outputs[0],val_labels[0]).cpu().detach())
      steps+=1

    # aggregate the final mean dice result
    metric = dice_metric.aggregate().item()
    print("DICE Metric:")
    print(metric)
    print("Averge Recall:")
    print(recall_total/steps)
    print("Average Jaccard:")
    print(jaccard_total/steps)

    # reset the status for next validation round
    dice_metric.reset()

if __name__ == "__main__":

    device_id = 1
    cuda_id = "cuda:" + str(device_id)
    device = torch.device(cuda_id)
    torch.cuda.set_device(cuda_id)
    dataset = "BRATS"
    # HEMIS_v1 is full UNet post fusion
    # HEMIS_v2 is 3 res_units post fusion, 4 outputs per modality
    # HEMIS_v3 is 3 res_units post fusion, 16 outputs per modality

    # net_type = "HEMIS_v2"
    # net_type = "UNet"
    # net_type = "UNet"
    net_type = "UNet"
    # net_type = "HEM spatial attention"
    # net_type = "MSFN"

    # net_path = "/home/sedm6251/projectMaterial/baseline_models/BRATS_Decathlon_2016_17/Training_Experiments/HEMIS_tests/HEMIS_trained_on_BRATS_and_ATLAS_full"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/ATLAS/UNet_trained_on_BRATS_and_ATLAS_full.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/ATLAS/single_UNet_Atlas.pth"
    # net_path = "/home/sedm6251/projectMaterial/Modality_Dropout_tests/uniform_modality_dropout.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/BRATS_Decathlon_2016_17/HEM_MEAN_BRATS_FLAIR_T1_T1gd_RAND_initial.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/BRATS_Decathlon_2016_17/HEM_Spatial_Attention_BRATS_RAND.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/BRATS_Decathlon_2016_17/UNETv2_BRATS_RAND_2ChannelLabels.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/BRATS_Decathlon_2016_17/UNETv2_BRATS_RAND_NEW_2ChannelSeg.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/UNET/RAND/UNetv2_BRATS_ATLAS_ISLES_RAND_BEST_BRATS.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/HEM_Spatial_Attention_BRATS_ATLAS_MSSEG_RAND_BEST_BRATS.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/HEM_Spatial_Attention_BRATS_ATLAS_MSSEG_RAND_AUGMENTED_BEST_ATLAS.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/UNET_BRATS_ATLAS_MSSEG_RAND_Epoch_449.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/MSFN_BRATS_RAND_Epoch_449.pth"


    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_MSSEG/HEM_SPATIAL_ATTENTION/RAND/HEM_Spatial_Attention_MSSEG_RAND_BEST_MSSEG.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_MSSEG/UNET/RAND/UNET_MSSEG_RAND_BEST_MSSEG.pth"

    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/UNET/RAND/UNetv2_BRATS_ATLAS_ISLES_RAND_BEST_ATLAS.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/UNET/RAND/UNetv2_BRATS_ATLAS_ISLES_RAND_BEST_BRATS.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/UNET/RAND/UNetv2_BRATS_ATLAS_ISLES_RAND_BEST_ISLES.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/UNET/RAND/UNetv2_BRATS_ATLAS_ISLES_RAND_Epoch_599.pth"

    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/HEM_SPATIAL_ATTENTION/HEM_Spatial_Attention_BRATS_ATLAS_ISLES_RAND_BEST_ATLAS.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/HEM_SPATIAL_ATTENTION/HEM_Spatial_Attention_BRATS_ATLAS_ISLES_RAND_BEST_BRATS.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/HEM_SPATIAL_ATTENTION/HEM_Spatial_Attention_BRATS_ATLAS_ISLES_RAND_BEST_ISLES.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/HEM_SPATIAL_ATTENTION/HEM_Spatial_Attention_BRATS_ATLAS_ISLES_RAND_Epoch_449.pth"

    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/HEM_SPATIAL_ATTENTION_AUGMENTED/HEM_Spatial_Attention_BRATS_ATLAS_ISLES_RAND_AUGMENTED_BEST_ATLAS.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/HEM_SPATIAL_ATTENTION_AUGMENTED/HEM_Spatial_Attention_BRATS_ATLAS_ISLES_RAND_AUGMENTED_BEST_BRATS.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/HEM_SPATIAL_ATTENTION_AUGMENTED/HEM_Spatial_Attention_BRATS_ATLAS_ISLES_RAND_AUGMENTED_BEST_ISLES.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/HEM_SPATIAL_ATTENTION_AUGMENTED/HEM_Spatial_Attention_BRATS_ATLAS_ISLES_RAND_AUGMENTED_Epoch_449.pth"

    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/UNET/RAND/UNET_BRATS_ATLAS_MSSEG_RAND_BEST_ATLAS.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/UNET/RAND/UNET_BRATS_ATLAS_MSSEG_RAND_BEST_BRATS.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/UNET/RAND/UNET_BRATS_ATLAS_MSSEG_RAND_BEST_MSSEG.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/UNET/RAND/UNET_BRATS_ATLAS_MSSEG_RAND_Epoch_449.pth"
    
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/HEM_SPATIAL_ATTENTION/RAND/HEM_Spatial_Attention_BRATS_ATLAS_MSSEG_RAND_BEST_ATLAS.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/HEM_SPATIAL_ATTENTION/RAND/HEM_Spatial_Attention_BRATS_ATLAS_MSSEG_RAND_BEST_BRATS.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/HEM_SPATIAL_ATTENTION/RAND/HEM_Spatial_Attention_BRATS_ATLAS_MSSEG_RAND_BEST_MSSEG.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/HEM_SPATIAL_ATTENTION/RAND/HEM_Spatial_Attention_BRATS_ATLAS_MSSEG_RAND_Epoch_499.pth"
    
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/HEM_SPATIAL_ATTENTION_AUGMENTED/HEM_Spatial_Attention_BRATS_ATLAS_MSSEG_RAND_AUGMENTED_BEST_ATLAS.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/HEM_SPATIAL_ATTENTION_AUGMENTED/HEM_Spatial_Attention_BRATS_ATLAS_MSSEG_RAND_AUGMENTED_BEST_BRATS.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/HEM_SPATIAL_ATTENTION_AUGMENTED/HEM_Spatial_Attention_BRATS_ATLAS_MSSEG_RAND_AUGMENTED_BEST_MSSEG.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/HEM_SPATIAL_ATTENTION_AUGMENTED/HEM_Spatial_Attention_BRATS_ATLAS_MSSEG_RAND_AUGMENTED_Epoch_449.pth"
    
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/MSFN/MSFN_BRATS_ATLAS_MSSEG_RAND_BEST_ATLAS.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/MSFN/MSFN_BRATS_ATLAS_MSSEG_RAND_BEST_BRATS.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/MSFN/MSFN_BRATS_ATLAS_MSSEG_RAND_BEST_MSSEG.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/MSFN/MSFN_BRATS_ATLAS_MSSEG_RAND_Epoch_999.pth"

    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ISLES_MSSEG/HEM_SPATIAL_ATTENTION/HEM_Spatial_Attention_BRATS_ISLES_MSSEG_BEST_BRATS.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ISLES_MSSEG/HEM_SPATIAL_ATTENTION/HEM_Spatial_Attention_BRATS_ISLES_MSSEG_BEST_ISLES.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ISLES_MSSEG/HEM_SPATIAL_ATTENTION/HEM_Spatial_Attention_BRATS_ISLES_MSSEG_BEST_MSSEG.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ISLES_MSSEG/HEM_SPATIAL_ATTENTION/HEM_Spatial_Attention_BRATS_ISLES_MSSEG_Epoch_449.pth"
    
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ISLES_MSSEG/UNET/UNET_BRATS_ISLES_MSSEG_RAND_BEST_BRATS.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ISLES_MSSEG/UNET/UNET_BRATS_ISLES_MSSEG_RAND_BEST_ISLES.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ISLES_MSSEG/UNET/UNET_BRATS_ISLES_MSSEG_RAND_BEST_MSSEG.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ISLES_MSSEG/UNET/UNET_BRATS_ISLES_MSSEG_RAND_Epoch_449.pth"

    net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ATLAS_ISLES_MSSEG/UNET/UNET_ATLAS_ISLES_MSSEG_RAND_BEST_ATLAS.pth"
    net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ATLAS_ISLES_MSSEG/UNET/UNET_ATLAS_ISLES_MSSEG_RAND_BEST_ISLES.pth"
    net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ATLAS_ISLES_MSSEG/UNET/UNET_ATLAS_ISLES_MSSEG_RAND_BEST_MSSEG.pth"
    # net_path = "/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ATLAS_ISLES_MSSEG/UNET/UNET_ATLAS_ISLES_MSSEG_RAND_Epoch_999.pth"

    # modalities to be tested
    modalities = [0,1,2,3]
    # map of channels to corresponding UNET inputs
    channel_map = [2,3,4,5]
    # modalities that were trained on by the UNet - if a modality is in the test set but wasn't trained on, it is excluded - can stay same for tests
    # on same datsetsf
    modalities_present_at_training = [0,1,2,3]

    modalities_trained_on = 6

    # res_units = 2
    
    #  only relevant for UNet
    # randomly_zero_modalities = False

    modalities_combinations = create_modality_combinations(modalities)
    model = create_net(net_path=net_path, net_type=net_type, device=device, modalities_trained_on=modalities_trained_on, cuda_id=cuda_id)
    dataloader = create_dataset(dataset=dataset)

    for combination in modalities_combinations:
      print(combination)
      test(model=model, val_loader=dataloader,modalities=combination,device=device, channel_map=channel_map, modalities_present_at_training=modalities_present_at_training,save_outputs=False,save_path="")