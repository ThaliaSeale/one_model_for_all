from monai.data import ImageDataset, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import Activations, EnsureChannelFirst, AsDiscrete, Compose, RandRotate90, RandSpatialCrop, ScaleIntensity, CenterSpatialCrop
import numpy as np
import torch
from torchmetrics.classification import Recall
import utils
import blob_detection as blob

def test(model, val_loader, dataset_name, modalities, net, device, modalities_present_at_training, save_outputs, save_path="/home/sedm2651/", detect_blobs=False):
  cropped_input_size = [128,128,128]
  dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
  post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

  with torch.no_grad():
    recall = Recall('binary').to(device)
    recall_total = 0.

    val_images = None
    val_labels = None
    val_outputs = None
    steps = 0

    dice_metric.reset()

    detected = []
    undetected = []
    det_seg_sizes = []
    undet_seg_sizes = []
    dice_metrics = []
    segment_pixel_vol = []
    gt_pixel_vol = []
    segmented_num_over_thresh = 0
    segmented_are_lesions = 0

    for val_data in val_loader:

      
      roi_size = (cropped_input_size[0], cropped_input_size[1], cropped_input_size[2])
      sw_batch_size = 1

      if net.net_type == "UNet" or net.net_type == "UNetv2":
        val_data[0] = utils.create_UNET_input(val_data, modalities, dataset_name, net, modalities_present_at_training)
        # val_data[0] = val_data[0][:,[0,1,2,3]]        
      else:
        val_data[0] = val_data[0][:,modalities,:,:,:]

        val_images, val_labels = val_data[0].to(device), val_data[1].to(device)

        val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
      
      val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]

      if save_outputs:
        file_save_path = save_path + str(steps) + ".nii.gz"
        utils.save_nifti(val_outputs[0], file_save_path)

      # compute metric for current iteration
      current_dice = dice_metric(y_pred=val_outputs, y=val_labels)
      # dice_metrics.append(float(current_dice.cpu().detach()))
      pixels_segmented = np.count_nonzero(val_outputs[0])
      # print(pixels_segmented)
      gt_segmented = np.count_nonzero(val_labels[0])
      # print("Segmtentation results:")
      # print(current_dice)
      # print(pixels_segmented)
      # print(gt_segmented)
      segment_pixel_vol.append(pixels_segmented)
      gt_pixel_vol.append(gt_segmented)


      if detect_blobs:
        [det, undet, det_seg_size, undet_seg_size] = blob.detect_blobs(val_labels[0], val_outputs[0])
        [out_segmented_are_lesions,out_segmented_num_over_thresh] = blob.count_over_thresh(val_labels[0],val_outputs[0])
        detected = detected + det
        undetected = undetected + undet
        det_seg_sizes = det_seg_sizes + det_seg_size
        undet_seg_sizes = undet_seg_sizes + undet_seg_size
        segmented_are_lesions += out_segmented_are_lesions
        segmented_num_over_thresh += out_segmented_num_over_thresh

      # print(float(current_dice))
      current_recall = float(recall(val_outputs[0],val_labels[0]).cpu().detach())
      recall_total += current_recall
      # print(current_recall)
      steps+=1

    # aggregate the final mean dice result
    metric = dice_metric.aggregate().item()

    # print("DICE Metric:")
    # print(metric)
    
    # print("Averge Recall:")
    # print(recall_total/steps)

    print("Num comps in seg over thresh ", segmented_num_over_thresh)
    print("Num comps over thresh are lesions ", segmented_are_lesions)
 
    if detect_blobs:
      blob.plot_hist(detected, undetected)

    # print("Mean pixels segmented:")
    # print(np.mean(segment_pixel_vol))

    # reset the status for next validation round
    dice_metric.reset()
    return dice_metrics, segment_pixel_vol, gt_pixel_vol, detected, undetected, det_seg_sizes, undet_seg_sizes
