import monai
from monai.networks.blocks import ResidualUnit
import torch

class Hemis_Net:

    def __init__(self, device):
        self.device = device

        self.model_1 = monai.networks.nets.UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=4,
            kernel_size = (3,3,3),
            channels=(16, 32, 64, 128, 256),
            strides=((2,2,2),(2,2,2),(2,2,2),(2,2,2)),
            num_res_units=2,
            dropout=0.2,
        ).to(device)

        self.model_2 = monai.networks.nets.UNet(
            spatial_dims=3,
            in_channels=8,
            out_channels=1,
            kernel_size = (3,3,3),
            channels=(16, 32, 64, 128, 256),
            strides=((2,2,2),(2,2,2),(2,2,2),(2,2,2)),
            num_res_units=2,
            dropout=0.2,
        ).to(device)

        self.parameters = list(self.model_1.parameters()) + list(self.model_2.parameters())


    def combined_model(self,batch_data):
        outs = []
        number_of_modalities = len(batch_data[0])
        # print("modalities_received")
        # print(number_of_modalities)

        for modality_index in range(number_of_modalities):
            modality_input = (batch_data[:,[modality_index],:,:,:]).to(self.device)
            modality_out = self.model_1(modality_input)
            outs.append(modality_out)


        if number_of_modalities != 1:
            outputs_tensor = torch.stack(outs).to(self.device)
            # print(outputs_tensor.shape)
            means = torch.mean(outputs_tensor,dim=0).to(self.device)
            # print(means.shape)
            vars = torch.var(outputs_tensor,dim=0).to(self.device)
            # print(vars.shape)
        else:
            means = outs[0]
            vars= torch.mul(outs[0],0.)

        final_input = torch.cat((means,vars),dim=1).to(self.device)
        final_outputs = self.model_2(final_input)

        return final_outputs

    def eval(self):
        self.model_1.eval()
        self.model_2.eval()

    def train(self):
        self.model_1.train()
        self.model_2.train()

    def load(self,load_path):
        initial_model_path = load_path + "_initial.pth"
        output_model_path = load_path + "_output.pth"

        cuda_id = str(self.device)
        self.model_1.load_state_dict(torch.load(initial_model_path,map_location={"cuda:0":cuda_id,"cuda:1":cuda_id}))
        self.model_2.load_state_dict(torch.load(output_model_path,map_location={"cuda:0":cuda_id,"cuda:1":cuda_id}))

