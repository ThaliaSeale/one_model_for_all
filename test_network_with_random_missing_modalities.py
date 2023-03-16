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
from Nets.HEMISv2 import HEMISv2 
from Nets.UNetv2 import UNetv2
from Nets.multi_scale_fusion_net import MSFN
import nibabel as nib
import epistemic_unc
import modality_uncertainty
import ensemble_unc
from glob import glob
import utils
# from Net_to_test import Net, EnsembleNet

def test(model, val_loader, dataset_name, modalities, net, device, modalities_present_at_training, save_outputs, save_path="/home/sedm2651/"):
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

    dice_metric.reset()
    for val_data in val_loader:

      
      
      roi_size = (cropped_input_size[0], cropped_input_size[1], cropped_input_size[2])
      sw_batch_size = 1

      # ep = epistemic_unc.Epistemic()
      # mean_out, var_mask = ep.calculate_unc(model, val_images)
      # val_outputs = mean_out

      # modality_unc = modality_uncertainty.Modality_Unc(model_type=net.net_type)
      # mean_out, var_mask, var = modality_unc.calculate_unc(model, val_images)

      if net.ensemble:
        ensemble_uncertainty = ensemble_unc.Ensemble_unc()
        mean_out, var_mask, var = ensemble_uncertainty.calculate_unc(model, val_data, dataset_name, modalities, modalities_present_at_training, device)
        val_labels = val_data[1].to(device)
        val_outputs = mean_out
      else:
        if net.net_type == "UNet":
          val_data[0] = utils.create_UNET_input(val_data, modalities, dataset_name, net, modalities_present_at_training)        
        else:
          val_data[0] = val_data[0][:,modalities,:,:,:]

        val_images, val_labels = val_data[0].to(device), val_data[1].to(device)

        # modality_unc = modality_uncertainty.Modality_Unc(model_type=net.net_type)
        # mean_out, var_mask, var = modality_unc.calculate_unc(model, val_images)
        # val_outputs = mean_out
        val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
      
      val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]

      # val_outputs = torch.mul(val_outputs[0],var_mask)

      if True:
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
        # print("MANUALLY REMOVING DWI")
        pass

      if save_outputs:
        file_save_path = save_path + str(steps) + ".nii.gz"
        utils.save_nifti(val_outputs[0], file_save_path)

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
