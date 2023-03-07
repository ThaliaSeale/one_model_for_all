import torch
# import torch.nn as nn
from monai.inferers import sliding_window_inference
from matplotlib import pyplot as plt
import nibabel as nib
import numpy as np
import utils
# from torchmetrics.functional import kl_divergence

class Modality_Unc:
    def __init__(self, model_type) -> None:
        self.model_type = model_type

    def calculate_unc(self, model, data):
        roi_size = [128,128,128]
        # data shape [batch, modalities, dim0, dim1, dim2]
        num_modalitities = data.shape[1]

        modalities = np.arange(num_modalitities)
        # We start with the output using all modalities
        outputs = [sliding_window_inference(data, roi_size, 1, model)]
        # outputs = []
        all_combinations = utils.create_modality_combinations(list(modalities))
        for m in range(num_modalitities):
            _data = data.clone().detach()
            if self.model_type == "UNET":
                _data[:,m] = 0.
            else:
                modalities_to_keep = list(modalities[:m]) + list(modalities[m+1:])
                _data = _data[:,modalities_to_keep]
            output = sliding_window_inference(_data, roi_size, 1, model)
            outputs.append(output)

        # for m in all_combinations:
        #     _data = data[:,m]
        #     output = sliding_window_inference(_data, roi_size, 1, model)
        #     outputs.append(output)

        # for m in range(num_modalitities):
        #     _data = data[:,[m]]
        #     output = sliding_window_inference(_data, roi_size, 1, model)
        #     outputs.append(output)


        stacked_outputs = torch.stack(outputs)
        probs = torch.stack([torch.sigmoid(out) for out in outputs])
        # normed = torch.stack([torch.div(T - torch.mean(T), torch.std(T)) for T in outputs])

        true_model_output = sliding_window_inference(data, roi_size, 1, model)
        mean_output = torch.mean(stacked_outputs,dim=0)

        var = torch.var(probs,dim=0)
        # mean_of_unc = torch.mean(var)
        # std_of_unc = torch.std(var)
        # var = torch.div(var-mean_of_unc,std_of_unc)
        # var = torch.div(var, torch.sum(var))
        var_mask = var<0.2
        

        # segmented_pixels_true = torch.sigmoid(true_model_output)>0.5
        # segmented_pixels_mean = torch.sigmoid(mean_output)>0.5
        # uncertain_pixels = var<0.2

        # segmented_pixels_mean = torch.mul(segmented_pixels_mean ,var_mask)

        # num_uncertain_pixels = torch.sum(uncertain_pixels).cpu().detach().numpy()
        # num_uncertain_predicted_pixels = torch.sum(torch.mul(uncertain_pixels,segmented_pixels_true)).cpu().detach().numpy()
        # num_predicted_pixels = torch.sum(segmented_pixels_true).cpu().detach().numpy()
        # percent_uncertain_predicted = 100 * num_uncertain_predicted_pixels/num_predicted_pixels
        # print("Num uncertain: ", num_uncertain_pixels)
        # print("Num uncertain predicted pixels: ", num_uncertain_predicted_pixels)
        # print("Num predicted: ", num_predicted_pixels)
        # print("Percent of predicted pixels uncertain: " + str(percent_uncertain_predicted) + "%")

        # tensors_to_plot = [data, var, var_mask, segmented_pixels_true, segmented_pixels_mean,
        #                    data, var, var_mask, segmented_pixels_true, segmented_pixels_mean,
        #                    data, var, var_mask, segmented_pixels_true, segmented_pixels_mean]
        # slices = [40, 40, 40, 40, 40,
        #           70, 70, 70, 70, 70,
        #           100, 100, 100, 100, 100]

        # utils.plot_slices(tensors_to_plot,slices,5)

        mean_output = torch.mul(mean_output, var_mask)
        return mean_output, var_mask, var
        print("Done")