import torch
# import torch.nn as nn
from monai.inferers import sliding_window_inference
from matplotlib import pyplot as plt
import nibabel as nib
# from torchmetrics.functional import kl_divergence

class Epistemic:
    def __init__(self) -> None:
        pass

    def enable_dropout(self, m):
        for each_module in m.modules():
            if each_module.__class__.__name__.startswith('Dropout'):
                each_module.train()

    def calculate_unc(self, model, data, n_trials = 10):
        roi_size = [128,128,128]
        # val_outputs = sliding_window_inference(data, roi_size, 1, model)

        model.eval()
        self.enable_dropout(model)

        outputs = [sliding_window_inference(data, roi_size, 1, model) for _ in range(n_trials)]
        stacked_outputs = torch.stack(outputs)

        probs = torch.stack([torch.sigmoid(out) for out in outputs])
        
        # probs = torch.div(stacked, torch.sum(stacked,dim=0))
        mean_output = torch.mean(stacked_outputs,dim=0)
        var = torch.var(probs,dim=0)
        var_mask = var<0.1

        segmented_pixels = mean_output>0.5
        uncertain_pixels = var>0.

        num_uncertain_pixels = torch.sum(uncertain_pixels).cpu().detach().numpy()
        num_uncertain_predicted_pixels = torch.sum(torch.mul(uncertain_pixels,segmented_pixels)).cpu().detach().numpy()
        num_predicted_pixels = torch.sum(segmented_pixels).cpu().detach().numpy()
        percent_uncertain_predicted = 100 * num_uncertain_predicted_pixels/num_predicted_pixels
        print("Num uncertain: ", num_uncertain_pixels)
        print("Num uncertain predicted pixels: ", num_uncertain_predicted_pixels)
        print("Num predicted: ", num_predicted_pixels)
        print("Percent of predicted pixels uncertain: " + str(percent_uncertain_predicted) + "%")


        

        # var_detached = var.cpu().detach().numpy()
        # # outputs_1_detached = outputs[0].cpu().detach().numpy()
        # mean_output_detached = mean_output.cpu().detach().numpy()
        # segmented = mean_output_detached>0.5
        # # prob_0_detached = probs[0].cpu().detach().numpy()
        # # prob_1_detached = probs[1].cpu().detach().numpy()
        # # prob_2_detached = probs[2].cpu().detach().numpy()

        # _, axs = plt.subplots(3,3)
        # axs[0,1].imshow(var_detached[0,0,:,:,70],cmap="gray")
        # axs[0,0].imshow(mean_output_detached[0,0,:,:,70],cmap="gray")
        # axs[0,2].imshow(segmented[0,0,:,:,70],cmap="gray")

        # axs[1,0].imshow(mean_output_detached[0,0,:,:,40],cmap="gray")
        # axs[1,1].imshow(var_detached[0,0,:,:,40],cmap="gray")
        # axs[1,2].imshow(segmented[0,0,:,:,40],cmap="gray")

        # axs[2,0].imshow(mean_output_detached[0,0,:,:,100],cmap="gray")
        # axs[2,1].imshow(var_detached[0,0,:,:,100],cmap="gray")
        # axs[2,2].imshow(segmented[0,0,:,:,100],cmap="gray")
        # # # axs[1,0].imshow(outputs_1_detached[0,0,:,:,outputs_1_detached.shape[2]//2 - 10],cmap="gray")
        # # # axs[1,1].imshow(prob_0_detached[0,0,:,:,prob_0_detached.shape[2]//2 - 10],cmap="gray")
        # # # axs[2,0].imshow(prob_1_detached[0,0,:,:,prob_1_detached.shape[2]//2 - 10],cmap="gray")
        # # # axs[2,1].imshow(prob_2_detached[0,0,:,:,prob_2_detached.shape[2]//2 - 10],cmap="gray")

        # plt.show()
        return mean_output, var_mask, var
        print("Done")