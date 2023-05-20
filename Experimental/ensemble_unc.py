import torch
# import torch.nn as nn
from monai.inferers import sliding_window_inference
from matplotlib import pyplot as plt
import nibabel as nib
import numpy as np
import utils
# from torchmetrics.functional import kl_divergence

class Ensemble_unc:
    def __init__(self) -> None:
        pass

    def calculate_unc(self, ensemble_model, data, dataset_name, modalities, modalities_present_at_training, device):
        roi_size = [128,128,128]
        # data shape [batch, modalities, dim0, dim1, dim2]
        descriptors = ensemble_model.net_descriptors
        models = ensemble_model.nets

        outputs = []
        for i, model in enumerate(models):
            if descriptors[i].net_type == "UNet":
                _data = utils.create_UNET_input(data, modalities, dataset_name, descriptors[i], modalities_present_at_training)        
            else:
                _data = data[0][:,modalities,:,:,:]

            val_images = _data.to(device)
            output = sliding_window_inference(val_images, roi_size, 1, model)
            outputs.append(output)


        stacked_outputs = torch.stack(outputs)
        probs = torch.stack([torch.sigmoid(out) for out in outputs])

        mean_output = torch.mean(stacked_outputs,dim=0)

        var = torch.var(probs,dim=0)
        var_mask = var>5.

        segmented_pixels_mean = torch.sigmoid(mean_output)>0.5
        uncertain_pixels = var>5.

        num_uncertain_pixels = torch.sum(uncertain_pixels).cpu().detach().numpy()
        num_uncertain_predicted_pixels = torch.sum(torch.mul(uncertain_pixels,segmented_pixels_mean)).cpu().detach().numpy()
        num_predicted_pixels = torch.sum(segmented_pixels_mean).cpu().detach().numpy()
        percent_uncertain_predicted = 100 * num_uncertain_predicted_pixels/num_predicted_pixels
        print("Num uncertain: ", num_uncertain_pixels)
        print("Num uncertain predicted pixels: ", num_uncertain_predicted_pixels)
        print("Num predicted: ", num_predicted_pixels)
        print("Percent of predicted pixels uncertain: " + str(percent_uncertain_predicted) + "%")

        tensors_to_plot = [data[0], var, var_mask, segmented_pixels_mean,
                           data[0], var, var_mask, segmented_pixels_mean,
                           data[0], var, var_mask, segmented_pixels_mean]
        slices = [40, 40, 40, 40, 40,
                  70, 70, 70, 70, 70,
                  100, 100, 100, 100, 100]

        # utils.plot_slices(tensors_to_plot,slices,4)

        return mean_output, var_mask, var
        print("Done")