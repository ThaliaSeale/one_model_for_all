import monai
# from monai.networks.blocks import ResidualUnit, Convolution
import torch
import torch.nn as nn
from Nets.UNetv2 import UNetv2
# import UNetv2
from Nets.updated_res_units import ResidualUnit, Convolution
# import monai.networks.nets.unet as UNet
# import orig_UNet_but_with_avg_pool_res_unit as UNetAvgPool
from Nets.UNet_block import UNet_block
from Nets.UNet_block_deprecated import UNet_block_deprecated
import nibabel as nib
import numpy as np
from Nets.theory_UNET import theory_UNET


class Channel_Attention(nn.Module):
    def __init__(self,
    ) -> None:
        super().__init__()

        self.glob_avg_pool = nn.AdaptiveAvgPool3d(output_size=1)
        # self.glob_max_pool = nn.AdaptiveMaxPool3d(output_size=1)
        softmax = nn.Softmax(dim=0)

        self.attention = nn.Sequential(self.glob_avg_pool, softmax)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att = self.attention(x)
        m = torch.mul(att, x)
        y = torch.sum(m,dim=0)
        return y


class Spatial_Attention(nn.Module):
    def __init__(self,
        in_channels: int,
        conv_1_out_channels: int,
        out_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()

        print("Using spatial attention")
        print("Channels in: ", in_channels)
        print("Mid channels: ", conv_1_out_channels)
        print("Out channels: ", out_channels)

        conv_1 = Convolution(spatial_dims=3, in_channels=in_channels, out_channels=conv_1_out_channels, kernel_size=3, strides=1,dropout=dropout)
        conv_2 = Convolution(spatial_dims=3, in_channels=conv_1_out_channels, out_channels=out_channels, kernel_size=3, strides=1, dropout=dropout)

        self.conv = nn.Sequential(conv_1,conv_2)
        self.softmax = nn.Softmax(dim=0)

    def save_imgs(self, img: torch.Tensor, name: str):
        # for i in range(img.shape[2]):
        for i in range(img.shape[2]):
            # modality, batch, channel, 128,128,128
            outputs_numpy = img[:,:,i,:,:,:].cpu().detach().numpy()
            outputs_numpy = np.squeeze(outputs_numpy)
            outputs_numpy = np.transpose(outputs_numpy,(1,2,3,0))
            new_image = nib.Nifti1Image(outputs_numpy,affine=None)
            save_path = "/home/sedm6251/spatial_att_outs/" + name + "_chan_" + str(i) + ".nii.gz"
            nib.save(new_image, save_path)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        
        modalities = torch.unbind(input=x,dim=0)
        for ind, modality in enumerate(modalities):
            if ind == 0:
                conv_outs = self.conv(modality)
            elif ind == 1:
                # print("WARNING BAD HEM IN USE")
                conv_outs = torch.stack((conv_outs, self.conv(modality)),dim=0)
            else:
                conv_outs = torch.cat((conv_outs, torch.unsqueeze(self.conv(modality),dim=0)),dim=0)
            # conv_outs.append(self.conv(modality))
        
        del modalities
        # torch.cuda.empty_cache()
        
        conv_outs = self.softmax(conv_outs)
        
        # s = torch.sum(conv_outs,dim=0)
        # conv_outs = torch.div(conv_outs, s)

        # self.save_imgs(x, "orig")
        # self.save_imgs(conv_outs, "attention_mask")

        conv_outs = torch.mul(conv_outs, x)

        # self.save_imgs(conv_outs, "image_with-attention")

        del x
        # out = torch.sum(conv_outs,dim=0)


        # for i in range(out.shape[1]):
        #     outputs_numpy = out[:,i,:,:,:].cpu().detach().numpy()
        #     outputs_numpy = np.squeeze(outputs_numpy)
        #     new_image = nib.Nifti1Image(outputs_numpy,affine=None)
        #     save_path = "/home/sedm6251/spatial_att_outs/output" + "_chan_" + str(i) + ".nii.gz"
        #     nib.save(new_image, save_path)

        return torch.sum(conv_outs,dim=0)

class HEMISv2(nn.Module):

    def __init__(self,
        post_seg_res_units: bool, 
        fusion_type: str,
        UNet_outs: int,
        conv1_in: int,
        conv1_out: int,
        conv2_in: int,
        conv2_out: int,
        conv3_in: int,
        pred_uncertainty: bool,
        grid_UNet: bool,
    ) -> None:
        super().__init__()

        print("UNet_outs, conv1_in, conv1_out, conv2_in, conv2_out, conv3_in")
        print(UNet_outs)
        print(conv1_in)
        print(conv1_out)
        print(conv2_in)
        print(conv2_out)
        print(conv3_in)

        print("UNET layers:")
        print("Layers: 16,32,64,128,256")

        self.fusion_type = fusion_type
        print("Fusion Type: ", fusion_type)

        self.pred_uncertainty = pred_uncertainty
        print("Predict Uncertainty: ", pred_uncertainty)

        dropout = 0.2
        print("Dropout: ",dropout)
        if grid_UNet:
            print("Using Grid UNET - transposed conv then res unit with 1 subunit")
            self.model_1 = UNet_block_deprecated(
                spatial_dims=3,
                in_channels=1,
                out_channels=UNet_outs,
                kernel_size = (3,3,3),
                channels=(16, 32, 64, 128, 256),
                strides=((2,2,2),(2,2,2),(2,2,2),(2,2,2)),
                num_res_units=2,
                dropout=dropout,
                last_layer_adn=True
            )
        else:
            # print("Using non-grid UNet - upsample then one conv")
            # self.model_1 = UNet_block(
            #     spatial_dims=3,
            #     in_channels=1,
            #     out_channels=UNet_outs,
            #     kernel_size = (3,3,3),
            #     channels=(16, 32, 64, 128, 256),
            #     strides=((2,2,2),(2,2,2),(2,2,2),(2,2,2)),
            #     num_res_units=2,
            #     dropout=dropout,
            #     last_layer_adn=True
            # )
            print("USING THEORY UNET")
            self.model_1 = theory_UNET(in_channels=1, out_channels = UNet_outs,last_layer_conv_only = True)

        # takes bool input to have residual units or just convs post fusion
        def create_seg_stage(res_units:bool) -> nn.Module:
            if res_units:
                print("Using 3 residual units post fusion")
                #  for residual units at post-fusion layer
                conv1 = ResidualUnit(
                    spatial_dims=3,
                    in_channels=conv1_in,
                    out_channels=conv1_out,
                    dropout=dropout,
                )

                conv2 = ResidualUnit(
                    spatial_dims=3,
                    in_channels=conv2_in,
                    out_channels=conv2_out,
                    dropout=dropout,
                )
                conv3 = ResidualUnit(
                    spatial_dims=3,
                    in_channels=conv3_in,
                    out_channels=1,
                    dropout=dropout,
                    last_conv_only=True,
                )
            else:
                print("Using just 3 convolutions post fusion (not residual units")
                #  for residual units at post-fusion layer
                conv1 = Convolution(
                    spatial_dims=3,
                    in_channels=conv1_in,
                    out_channels=conv1_out,
                    dropout=dropout,
                )

                conv2 = Convolution(
                    spatial_dims=3,
                    in_channels=conv2_in,
                    out_channels=conv2_out,
                    dropout=dropout,
                )
                conv3 = Convolution(
                    spatial_dims=3,
                    in_channels=conv3_in,
                    out_channels=1,
                    dropout=dropout,
                    conv_only=True,
                )
            
            return nn.Sequential(conv1,conv2,conv3)
        
        def create_uncertainty_seg_stage():

            conv1 = Convolution(
                spatial_dims=3,
                in_channels=conv1_in,
                out_channels=conv1_out,
                dropout=dropout,
            )

            conv2 = Convolution(
                spatial_dims=3,
                in_channels=conv2_in,
                out_channels=conv2_out,
                dropout=dropout,
            )

            initial_convs = nn.Sequential(conv1,conv2)
            
            seg_head = Convolution(
                    spatial_dims=3,
                    in_channels=conv3_in,
                    out_channels=1,
                    dropout=dropout,
                    conv_only=True,
            )
            
            unc_head = Convolution(
                spatial_dims=3,
                in_channels=conv3_in,
                out_channels=1,
                dropout=dropout,
                conv_only=False,
                act="softplus",
                norm=None
            )
            return initial_convs, seg_head, unc_head


        if self.fusion_type == "channel attention":
            self.channel_attention_fusion = Channel_Attention()
        elif self.fusion_type == "spatial attention":
            self.spatial_attention_fusion = Spatial_Attention(in_channels=UNet_outs,conv_1_out_channels=7,out_channels=UNet_outs,dropout=dropout)

        if pred_uncertainty:
            self.initial_convs, self.seg_head, self.unc_head = create_uncertainty_seg_stage()
        else:
            self.seg_stage = create_seg_stage(post_seg_res_units)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        number_of_modalities = len(x[0])

        for modality_index in range(number_of_modalities):
            modality_input = (x[:,[modality_index],:,:,:])
            modality_out = self.model_1(modality_input)
            outs.append(modality_out)
        
        outputs_tensor = torch.stack(outs)
        del outs
        del modality_input
        del modality_out
        # torch.cuda.empty_cache()

        if self.fusion_type == "mean":
            final_input = torch.mean(outputs_tensor,dim=0) 
        elif self.fusion_type == "mean var":
            if number_of_modalities == 1:
                outputs_tensor = torch.squeeze(outputs_tensor, dim=0)
                means = outputs_tensor
                vars = torch.mul(outputs_tensor, 0.)
            else:
                means = torch.mean(outputs_tensor,dim=0) 
                vars = torch.var(outputs_tensor,dim=0)
            final_input = torch.cat((means,vars),dim=1)
        elif self.fusion_type == "max":
            (final_input,_) = torch.max(outputs_tensor,dim=0)
        elif self.fusion_type == "mean max":
            means = torch.mean(outputs_tensor,dim=0) 
            (maxes,_) = torch.max(outputs_tensor,dim=0)
            final_input = torch.cat((means,maxes),dim=1)
        elif self.fusion_type == "max var":
            if number_of_modalities == 1:
                outputs_tensor = torch.squeeze(outputs_tensor, dim=0)
                maxes = outputs_tensor
                vars = torch.mul(outputs_tensor, 0.)
            else:
                (maxes,_) = torch.max(outputs_tensor,dim=0)
                vars = torch.var(outputs_tensor,dim=0)
            final_input = torch.cat((maxes,vars),dim=1)
        elif self.fusion_type == "channel attention":
            final_input = self.channel_attention_fusion(outputs_tensor)
        elif self.fusion_type =="spatial attention":
            final_input = self.spatial_attention_fusion(outputs_tensor)
        
        del outputs_tensor
        # torch.cuda.empty_cache()


        if self.pred_uncertainty:
            return self.seg_head(self.initial_convs(final_input)), self.unc_head(self.initial_convs)
        else:
            return self.seg_stage(final_input)





HEMISv2 = HEMISv2