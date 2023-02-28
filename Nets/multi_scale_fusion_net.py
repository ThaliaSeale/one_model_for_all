import monai
# from monai.networks.blocks import ResidualUnit, Convolution
import torch
import torch.nn as nn
from Nets.UNetv2 import UNetv2
# import UNetv2
from updated_res_units import ResidualUnit, Convolution
# import monai.networks.nets.unet as UNet
# import orig_UNet_but_with_avg_pool_res_unit as UNetAvgPool
from UNet_block import UNet_block
from UNet_block_deprecated import UNet_block_deprecated
import nibabel as nib
import numpy as np
from itertools import combinations
import random


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
                conv_outs = torch.stack((modality, self.conv(modality)),dim=0)
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

class Paired_Spatial_Attention(nn.Module):
    def __init__(self,
        in_channels: int,
        conv_1_out_channels: int,
        out_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()

        print("Using paired spatial attention")
        print("Channels in: ", 2*in_channels)
        print("Mid channels: ", 2*conv_1_out_channels)
        print("Out channels: ", 2*out_channels)

        self.in_channels = in_channels
        conv_1 = Convolution(spatial_dims=3, in_channels=2*in_channels, out_channels=2*conv_1_out_channels, kernel_size=3, strides=1,dropout=dropout)
        conv_2 = Convolution(spatial_dims=3, in_channels=2*conv_1_out_channels, out_channels=2*out_channels, kernel_size=3, strides=1, dropout=dropout)

        self.conv = nn.Sequential(conv_1,conv_2)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        num_modalities = x.shape[0]
        modality_combinations = list(combinations(np.arange(num_modalities),2))
        modalities = torch.unbind(input=x,dim=0)
        dict_masks = {}
        for combination in modality_combinations:
            combination = list(combination)
            # random.shuffle(combination)
            mod_1_index = combination[0]
            mod_2_index = combination[1]
            mod_1 = modalities[mod_1_index]
            mod_2 = modalities[mod_2_index]
            conv_input = torch.cat((mod_1,mod_2),dim=1)
            conv_out = self.conv(conv_input)
            mask_1 = conv_out[:,:self.in_channels]
            mask_2 = conv_out[:,-self.in_channels:]
            
            if mod_1_index in dict_masks:
                # if the entry in the dictionary does exist and the dimensionality is 5 (i.e. just batch x channel x dims), not the new dimension
                if len(dict_masks[mod_1_index].shape) == 5:
                    dict_masks[mod_1_index] = torch.stack((dict_masks[mod_1_index], mask_1),dim=0)
                else:
                    dict_masks[mod_1_index] = torch.cat((dict_masks[mod_1_index], torch.unsqueeze(mask_1,dim=0)), dim=0)
            else:
                dict_masks[mod_1_index] = mask_1

            if mod_2_index in dict_masks:
                # if the entry in the dictionary does exist and the dimensionality is 5 (i.e. just batch x channel x dims), not the new dimension
                if len(dict_masks[mod_2_index].shape) == 5:
                    dict_masks[mod_2_index] = torch.stack((dict_masks[mod_2_index], mask_2),dim=0)
                else:
                    dict_masks[mod_2_index] = torch.cat((dict_masks[mod_2_index], torch.unsqueeze(mask_2,dim=0)), dim=0)
            else:
                dict_masks[mod_2_index] = mask_2
        
        # tensor containinty sum of all the averaged tensors for normalisation
        # sum_tensor = torch.zeros_like(mask_1)
        for modality in dict_masks:
            mean = torch.mean(dict_masks[modality],dim=0)
            dict_masks[modality] = mean
            # sum_tensor = sum_tensor + mean

        att_mask_out = torch.zeros_like(x)
        for modality in dict_masks:
            # att_mask_out[modality] = torch.div(dict_masks[modality],sum_tensor)
            att_mask_out[modality] = dict_masks[modality]
        att_mask_out = self.softmax(att_mask_out)

        # outputs_numpy = att_mask_out[:,0,0,:,:,:].cpu().detach().numpy()
        # outputs_numpy = np.squeeze(outputs_numpy)
        # # outputs_numpy = np.transpose(outputs_numpy,(1,2,3,0))
        # new_image = nib.Nifti1Image(outputs_numpy,affine=None)
        # save_path = "/home/sedm6251/DELETE_mask.nii.gz"
        # nib.save(new_image, save_path)

        # outputs_numpy = x[:,0,0,:,:,:].cpu().detach().numpy()
        # outputs_numpy = np.squeeze(outputs_numpy)
        # # outputs_numpy = np.transpose(outputs_numpy,(1,2,3,0))
        # new_image = nib.Nifti1Image(outputs_numpy,affine=None)
        # save_path = "/home/sedm6251/DELETE_in.nii.gz"
        # nib.save(new_image, save_path)
        
        m = torch.mul(att_mask_out, x)

        # outputs_numpy = m[:,0,0,:,:,:].cpu().detach().numpy()
        # outputs_numpy = np.squeeze(outputs_numpy)
        # # outputs_numpy = np.transpose(outputs_numpy,(1,2,3,0))
        # new_image = nib.Nifti1Image(outputs_numpy,affine=None)
        # save_path = "/home/sedm6251/DELETE_after_mask.nii.gz"
        # nib.save(new_image, save_path)


        s = torch.sum(m,dim=0)

        # outputs_numpy = s[0,0,:,:,:].cpu().detach().numpy()
        # outputs_numpy = np.squeeze(outputs_numpy)
        # # outputs_numpy = np.transpose(outputs_numpy,(1,2,3,0))
        # new_image = nib.Nifti1Image(outputs_numpy,affine=None)
        # save_path = "/home/sedm6251/DELETE_fused.nii.gz"
        # nib.save(new_image, save_path)

        return s

        


class MSFN(nn.Module):

    def __init__(self,
        paired: bool
    ) -> None:
        super().__init__()

        dropout = 0.2
        print("Dropout: ",dropout)
        
        # self.spatial_attention_fusion_1 = Spatial_Attention(in_channels=16,conv_1_out_channels=8,out_channels=16,dropout=dropout)
        # self.spatial_attention_fusion_2 = Spatial_Attention(in_channels=32,conv_1_out_channels=16,out_channels=32,dropout=dropout)
        # self.spatial_attention_fusion_3 = Spatial_Attention(in_channels=64,conv_1_out_channels=32,out_channels=64,dropout=dropout)
        # self.spatial_attention_fusion_4 = Spatial_Attention(in_channels=128,conv_1_out_channels=64,out_channels=128,dropout=dropout)
        # self.spatial_attention_fusion_5 = Spatial_Attention(in_channels=256,conv_1_out_channels=128,out_channels=256,dropout=dropout)
        if paired:
            self.spatial_attention_fusion_1 = Paired_Spatial_Attention(in_channels=16,conv_1_out_channels=8,out_channels=16,dropout=dropout)
            self.spatial_attention_fusion_2 = Paired_Spatial_Attention(in_channels=32,conv_1_out_channels=16,out_channels=32,dropout=dropout)
            self.spatial_attention_fusion_3 = Paired_Spatial_Attention(in_channels=64,conv_1_out_channels=32,out_channels=64,dropout=dropout)
            self.spatial_attention_fusion_4 = Paired_Spatial_Attention(in_channels=128,conv_1_out_channels=64,out_channels=128,dropout=dropout)
            self.spatial_attention_fusion_5 = Paired_Spatial_Attention(in_channels=256,conv_1_out_channels=128,out_channels=256,dropout=dropout)
        else:
            self.spatial_attention_fusion_1 = Spatial_Attention(in_channels=16,conv_1_out_channels=8,out_channels=16,dropout=dropout)
            self.spatial_attention_fusion_2 = Spatial_Attention(in_channels=32,conv_1_out_channels=16,out_channels=32,dropout=dropout)
            self.spatial_attention_fusion_3 = Spatial_Attention(in_channels=64,conv_1_out_channels=32,out_channels=64,dropout=dropout)
            self.spatial_attention_fusion_4 = Spatial_Attention(in_channels=128,conv_1_out_channels=64,out_channels=128,dropout=dropout)
            self.spatial_attention_fusion_5 = Spatial_Attention(in_channels=256,conv_1_out_channels=128,out_channels=256,dropout=dropout)

        self.conv_1 = ResidualUnit(spatial_dims=3,in_channels=1,out_channels=16,strides=2,kernel_size=3,subunits=2,dropout=0.2)
        self.conv_2 = ResidualUnit(spatial_dims=3,in_channels=16,out_channels=32,strides=2,kernel_size=3,subunits=2,dropout=0.2)
        self.conv_3 = ResidualUnit(spatial_dims=3,in_channels=32,out_channels=64,strides=2,kernel_size=3,subunits=2,dropout=0.2)
        self.conv_4 = ResidualUnit(spatial_dims=3,in_channels=64,out_channels=128,strides=2,kernel_size=3,subunits=2,dropout=0.2)
        self.conv_5 = ResidualUnit(spatial_dims=3,in_channels=128,out_channels=256,strides=1,kernel_size=3,subunits=2,dropout=0.2)

        upsample = torch.nn.Upsample(scale_factor=2)

        up_conv_1_a = Convolution(spatial_dims=3,in_channels=384,out_channels=384,strides=1,kernel_size=3,dropout=0.2)
        up_conv_2_a = Convolution(spatial_dims=3,in_channels=128,out_channels=128,strides=1,kernel_size=3,dropout=0.2)
        up_conv_3_a = Convolution(spatial_dims=3,in_channels=64,out_channels=64,strides=1,kernel_size=3,dropout=0.2)
        up_conv_4_a = Convolution(spatial_dims=3,in_channels=32,out_channels=32,strides=1,kernel_size=3,dropout=0.2)

        up_conv_1_b = Convolution(spatial_dims=3,in_channels=384,out_channels=64,strides=1,kernel_size=3,dropout=0.2)
        up_conv_2_b = Convolution(spatial_dims=3,in_channels=128,out_channels=32,strides=1,kernel_size=3,dropout=0.2)
        up_conv_3_b = Convolution(spatial_dims=3,in_channels=64,out_channels=16,strides=1,kernel_size=3,dropout=0.2)
        up_conv_4_b = Convolution(spatial_dims=3,in_channels=32,out_channels=1,strides=1,kernel_size=3,dropout=0.2,conv_only=True)

        self.up_stage_1 = nn.Sequential(upsample, up_conv_1_a, up_conv_1_b)
        self.up_stage_2 = nn.Sequential(upsample, up_conv_2_a, up_conv_2_b)
        self.up_stage_3 = nn.Sequential(upsample, up_conv_3_a, up_conv_3_b)
        self.up_stage_4 = nn.Sequential(upsample, up_conv_4_a, up_conv_4_b)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs_1 = []
        outs_2 = []
        outs_3 = []
        outs_4 = []
        outs_5 = []

        number_of_modalities = len(x[0])

        for modality_index in range(number_of_modalities):
            modality_input = (x[:,[modality_index],:,:,:])

            modality_out_1 = self.conv_1(modality_input)
            modality_out_2 = self.conv_2(modality_out_1)
            modality_out_3 = self.conv_3(modality_out_2)
            modality_out_4 = self.conv_4(modality_out_3)
            modality_out_5 = self.conv_5(modality_out_4)

            outs_1.append(modality_out_1)
            outs_2.append(modality_out_2)
            outs_3.append(modality_out_3)
            outs_4.append(modality_out_4)
            outs_5.append(modality_out_5)
        
        outputs_tensor_1 = torch.stack(outs_1)
        outputs_tensor_2 = torch.stack(outs_2)
        outputs_tensor_3 = torch.stack(outs_3)
        outputs_tensor_4 = torch.stack(outs_4)
        outputs_tensor_5 = torch.stack(outs_5)

        fused_1 = self.spatial_attention_fusion_1(outputs_tensor_1)
        fused_2 = self.spatial_attention_fusion_2(outputs_tensor_2)
        fused_3 = self.spatial_attention_fusion_3(outputs_tensor_3)
        fused_4 = self.spatial_attention_fusion_4(outputs_tensor_4)
        fused_5 = self.spatial_attention_fusion_5(outputs_tensor_5)

        up_in_1 = torch.cat((fused_5,fused_4),dim=1)
        up_out_1 = self.up_stage_1(up_in_1)

        up_in_2 = torch.cat((up_out_1,fused_3),dim=1)
        up_out_2 = self.up_stage_2(up_in_2)

        up_in_3 = torch.cat((up_out_2,fused_2),dim=1)
        up_out_3 = self.up_stage_3(up_in_3)

        up_in_4 = torch.cat((up_out_3,fused_1),dim=1)
        up_out_4 = self.up_stage_4(up_in_4)


        # outputs_numpy = up_out_4[:,0,:,:,:].cpu().detach().numpy()
        # # outputs_numpy = np.squeeze(outputs_numpy)
        # outputs_numpy = np.transpose(outputs_numpy,(1,2,3,0))
        # new_image = nib.Nifti1Image(outputs_numpy,affine=None)
        # save_path = "/home/sedm6251/spatial_att_outs/out_4.nii.gz"
        # nib.save(new_image, save_path)

        # sigmoided = torch.sigmoid(up_out_4)
        # outputs_numpy = sigmoided[:,0,:,:,:].cpu().detach().numpy()
        # # outputs_numpy = np.squeeze(outputs_numpy)
        # outputs_numpy = np.transpose(outputs_numpy,(1,2,3,0))
        # new_image = nib.Nifti1Image(outputs_numpy,affine=None)
        # save_path = "/home/sedm6251/spatial_att_outs/out_4_sig.nii.gz"
        # nib.save(new_image, save_path)


        # final_out = self.up_stage_4(torch.cat(self.up_stage_3(torch.cat(self.up_stage_2(torch.cat(self.up_stage_1(torch.cat(fused_5,fused_4)),fused_3)),fused_2)),fused_1))

        return up_out_4


MSFN = MSFN