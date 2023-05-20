import test
import torch
from monai.utils import set_determinism
import utils
from Net_to_test import Net
import pandas as pd

if __name__=="__main__":

    nets = [

    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_MSSEG/HEM_SPATIAL_ATTENTION/RAND/HEM_Spatial_Attention_MSSEG_RAND_BEST_MSSEG.pth", "HEM spatial attention", 0, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_MSSEG/UNET/RAND/UNET_MSSEG_RAND_BEST_MSSEG.pth", "UNet", 5, {"BRATS":[1,2,3,4], "ATLAS":[2], "MSSEG":[1,2,3,4,0], "ISLES2015":[1,2,4,3]}),

    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/UNET/RAND/UNetv2_BRATS_ATLAS_ISLES_RAND_BEST_ATLAS.pth", "UNet", 5, {"BRATS":[1,2,3,4], "ATLAS":[2], "MSSEG":[1,2,3,4,0], "ISLES2015":[1,2,4,0], "TBI":[1,2,4,3], "WMH": [1,2]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/UNET/RAND/UNetv2_BRATS_ATLAS_ISLES_RAND_BEST_BRATS.pth", "UNet", 5, {"BRATS":[1,2,3,4], "ATLAS":[2], "MSSEG":[1,2,3,4,0], "ISLES2015":[1,2,4,0], "TBI":[1,2,4,3], "WMH": [1,2]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/UNET/RAND/UNetv2_BRATS_ATLAS_ISLES_RAND_BEST_ISLES.pth", "UNet", 5, {"BRATS":[1,2,3,4], "ATLAS":[2], "MSSEG":[1,2,3,4,0], "ISLES2015":[1,2,4,0], "TBI":[1,2,4,3], "WMH": [1,2]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/UNET/RAND/UNetv2_BRATS_ATLAS_ISLES_RAND_Epoch_599.pth", "UNet", 5, {"BRATS":[1,2,3,4], "ATLAS":[2], "MSSEG":[1,2,3,4,0], "ISLES2015":[1,2,4,0], "TBI":[1,2,4,3], "WMH": [1,2]}),

    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/HEM_SPATIAL_ATTENTION/HEM_Spatial_Attention_BRATS_ATLAS_ISLES_RAND_BEST_ATLAS.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/HEM_SPATIAL_ATTENTION/HEM_Spatial_Attention_BRATS_ATLAS_ISLES_RAND_BEST_BRATS.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/HEM_SPATIAL_ATTENTION/HEM_Spatial_Attention_BRATS_ATLAS_ISLES_RAND_BEST_ISLES.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/HEM_SPATIAL_ATTENTION/HEM_Spatial_Attention_BRATS_ATLAS_ISLES_RAND_Epoch_449.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),

    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/HEM_SPATIAL_ATTENTION_AUGMENTED/HEM_Spatial_Attention_BRATS_ATLAS_ISLES_RAND_AUGMENTED_BEST_ATLAS.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/HEM_SPATIAL_ATTENTION_AUGMENTED/HEM_Spatial_Attention_BRATS_ATLAS_ISLES_RAND_AUGMENTED_BEST_BRATS.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/HEM_SPATIAL_ATTENTION_AUGMENTED/HEM_Spatial_Attention_BRATS_ATLAS_ISLES_RAND_AUGMENTED_BEST_ISLES.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/HEM_SPATIAL_ATTENTION_AUGMENTED/HEM_Spatial_Attention_BRATS_ATLAS_ISLES_RAND_AUGMENTED_Epoch_449.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),

    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/MSFN/MSFN_BRATS_ATLAS_ISLES_RAND_BEST_ATLAS.pth", "MSFN", 0, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/MSFN/MSFN_BRATS_ATLAS_ISLES_RAND_BEST_BRATS.pth", "MSFN", 0, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/MSFN/MSFN_BRATS_ATLAS_ISLES_RAND_BEST_ISLES.pth", "MSFN", 0, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/MSFN/MSFN_BRATS_ATLAS_ISLES_RAND_Epoch_999.pth", "MSFN", 0, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),

    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/MSFN_PAIRED/MSFN_PAIRED_BRATS_ATLAS_ISLES_BEST_ATLAS.pth", "MSFNP",5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/MSFN_PAIRED/MSFN_PAIRED_BRATS_ATLAS_ISLES_BEST_BRATS.pth", "MSFNP",5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES/MSFN_PAIRED/MSFN_PAIRED_BRATS_ATLAS_ISLES_Epoch_449.pth", "MSFNP",5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),




    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/UNET/RAND/UNET_BRATS_ATLAS_MSSEG_RAND_BEST_ATLAS.pth", "UNet", 5, {"BRATS":[1,2,3,4], "ATLAS":[2], "MSSEG":[1,2,3,4,0], "ISLES2015":[1,2,4,3], "TBI":[1,2,4,3], "WMH":[1,2]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/UNET/RAND/UNET_BRATS_ATLAS_MSSEG_RAND_BEST_BRATS.pth", "UNet", 5, {"BRATS":[1,2,3,4], "ATLAS":[2], "MSSEG":[1,2,3,4,0], "ISLES2015":[1,2,4,3], "TBI":[1,2,4,3], "WMH":[1,2]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/UNET/RAND/UNET_BRATS_ATLAS_MSSEG_RAND_BEST_MSSEG.pth", "UNet", 5, {"BRATS":[1,2,3,4], "ATLAS":[2], "MSSEG":[1,2,3,4,0], "ISLES2015":[1,2,4,3], "TBI":[1,2,4,3], "WMH":[1,2]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/UNET/RAND/UNET_BRATS_ATLAS_MSSEG_RAND_Epoch_449.pth", "UNet", 5, {"BRATS":[1,2,3,4], "ATLAS":[2], "MSSEG":[1,2,3,4,0], "ISLES2015":[1,2,4,3], "TBI":[1,2,4,3], "WMH":[1,2]}),
    
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/HEM_SPATIAL_ATTENTION/RAND/HEM_Spatial_Attention_BRATS_ATLAS_MSSEG_RAND_BEST_ATLAS.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/HEM_SPATIAL_ATTENTION/RAND/HEM_Spatial_Attention_BRATS_ATLAS_MSSEG_RAND_BEST_BRATS.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/HEM_SPATIAL_ATTENTION/RAND/HEM_Spatial_Attention_BRATS_ATLAS_MSSEG_RAND_BEST_MSSEG.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/HEM_SPATIAL_ATTENTION/RAND/HEM_Spatial_Attention_BRATS_ATLAS_MSSEG_RAND_Epoch_499.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/HEM_SPATIAL_ATTENTION_AUGMENTED/HEM_Spatial_Attention_BRATS_ATLAS_MSSEG_RAND_AUGMENTED_BEST_ATLAS.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/HEM_SPATIAL_ATTENTION_AUGMENTED/HEM_Spatial_Attention_BRATS_ATLAS_MSSEG_RAND_AUGMENTED_BEST_BRATS.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/HEM_SPATIAL_ATTENTION_AUGMENTED/HEM_Spatial_Attention_BRATS_ATLAS_MSSEG_RAND_AUGMENTED_BEST_MSSEG.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/HEM_SPATIAL_ATTENTION_AUGMENTED/HEM_Spatial_Attention_BRATS_ATLAS_MSSEG_RAND_AUGMENTED_Epoch_449.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/MSFN/MSFN_BRATS_ATLAS_MSSEG_RAND_BEST_ATLAS.pth", "MSFN", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/MSFN/MSFN_BRATS_ATLAS_MSSEG_RAND_BEST_BRATS.pth", "MSFN", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/MSFN/MSFN_BRATS_ATLAS_MSSEG_RAND_BEST_MSSEG.pth", "MSFN", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/MSFN/MSFN_BRATS_ATLAS_MSSEG_RAND_Epoch_999.pth", "MSFN", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),

    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ISLES_MSSEG/HEM_SPATIAL_ATTENTION/HEM_Spatial_Attention_BRATS_ISLES_MSSEG_BEST_BRATS.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ISLES_MSSEG/HEM_SPATIAL_ATTENTION/HEM_Spatial_Attention_BRATS_ISLES_MSSEG_BEST_ISLES.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ISLES_MSSEG/HEM_SPATIAL_ATTENTION/HEM_Spatial_Attention_BRATS_ISLES_MSSEG_BEST_MSSEG.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ISLES_MSSEG/HEM_SPATIAL_ATTENTION/HEM_Spatial_Attention_BRATS_ISLES_MSSEG_Epoch_449.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ISLES_MSSEG/UNET/UNET_BRATS_ISLES_MSSEG_RAND_BEST_BRATS.pth", "UNet", 6, {"BRATS":[2,3,4,5], "ATLAS":[3], "MSSEG":[2,3,4,5,0], "ISLES2015":[2,3,5,1]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ISLES_MSSEG/UNET/UNET_BRATS_ISLES_MSSEG_RAND_BEST_ISLES.pth", "UNet", 6, {"BRATS":[2,3,4,5], "ATLAS":[3], "MSSEG":[2,3,4,5,0], "ISLES2015":[2,3,5,1]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ISLES_MSSEG/UNET/UNET_BRATS_ISLES_MSSEG_RAND_BEST_MSSEG.pth", "UNet", 6, {"BRATS":[2,3,4,5], "ATLAS":[3], "MSSEG":[2,3,4,5,0], "ISLES2015":[2,3,5,1]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ISLES_MSSEG/UNET/UNET_BRATS_ISLES_MSSEG_RAND_Epoch_449.pth", "UNet", 6, {"BRATS":[2,3,4,5], "ATLAS":[3], "MSSEG":[2,3,4,5,0], "ISLES2015":[2,3,5,1]}),

    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ATLAS_ISLES_MSSEG/UNET/UNET_ATLAS_ISLES_MSSEG_RAND_BEST_ATLAS.pth", "UNet", 6, {"BRATS":[2,3,4,5], "ATLAS":[3], "MSSEG":[2,3,4,5,0], "ISLES2015":[2,3,5,1]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ATLAS_ISLES_MSSEG/UNET/UNET_ATLAS_ISLES_MSSEG_RAND_BEST_ISLES.pth", "UNet", 6, {"BRATS":[2,3,4,5], "ATLAS":[3], "MSSEG":[2,3,4,5,0], "ISLES2015":[2,3,5,1]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ATLAS_ISLES_MSSEG/UNET/UNET_ATLAS_ISLES_MSSEG_RAND_BEST_MSSEG.pth", "UNet", 6, {"BRATS":[2,3,4,5], "ATLAS":[3], "MSSEG":[2,3,4,5,0], "ISLES2015":[2,3,5,1]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ATLAS_ISLES_MSSEG/UNET/UNET_ATLAS_ISLES_MSSEG_RAND_Epoch_999.pth", "UNet", 6, {"BRATS":[2,3,4,5], "ATLAS":[3], "MSSEG":[2,3,4,5,0], "ISLES2015":[2,3,5,1]})

    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/MSFN_PAIRED_BRATS_ALL_Epoch_199.pth", "MSFN",5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]})

    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/MSFNP/MSFN_PAIRED_BRATS_ATLAS_MSSEG_RAND_BEST_ATLAS.pth", "MSFNP",5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/MSFNP/MSFN_PAIRED_BRATS_ATLAS_MSSEG_RAND_BEST_BRATS.pth", "MSFNP",5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/MSFNP/MSFN_PAIRED_BRATS_ATLAS_MSSEG_RAND_BEST_MSSEG.pth", "MSFNP",5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/MSFNP/MSFN_PAIRED_BRATS_ATLAS_MSSEG_RAND_Epoch_499.pth", "MSFNP",5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),


    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS/UNET/UNET_BRATS_ATLAS_BEST_BRATS.pth", "UNet",4, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[0,1,3,2],"TBI":[0,1,3,2], "WMH":[0,1]}),
    
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS/HEM/HEM_BRATS_ATLAS_BEST_BRATS.pth", "HEM spatial attention",5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS/MSFN_PAIRED/MSFN_PAIRED_BRATS_ATLAS_BEST_BRATS.pth", "MSFNP",5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),


    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS/MSFN_PAIRED/MSFN_PAIRED_BRATS_ATLAS_BEST_ATLAS.pth", "MSFNP",5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS/MSFN_PAIRED/MSFN_PAIRED_BRATS_ATLAS_BEST_BRATS.pth", "MSFNP",5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS/MSFN_PAIRED/MSFN_PAIRED_BRATS_ATLAS_Epoch_499.pth", "MSFNP",5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),

    # EnsembleNet(
    #     [
    #     Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/UNET/RAND/UNET_BRATS_ATLAS_MSSEG_RAND_BEST_BRATS.pth", "UNet", 5, {"BRATS":[1,2,3,4], "ATLAS":[2], "MSSEG":[1,2,3,4,0], "ISLES2015":[1,2,4,3]}),
        # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/HEM_SPATIAL_ATTENTION/RAND/HEM_Spatial_Attention_BRATS_ATLAS_MSSEG_RAND_BEST_BRATS.pth", "HEM spatial attention", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    #     Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/MSFN/MSFN_BRATS_ATLAS_MSSEG_RAND_BEST_BRATS.pth", "MSFN", 5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    #     Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG/MSFNP/MSFN_PAIRED_BRATS_ATLAS_MSSEG_RAND_BEST_BRATS.pth", "MSFNP",5, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[]}),
    #     ]
    # )

    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG_TBI_WMH/UNET/UNET_BRATS_ATLAS_MSSEG_TBI_WMH_RAND_Epoch_199.pth", "UNetv2", 6, {"BRATS":[1,3,4,5], "ATLAS":[3], "MSSEG":[1,3,4,5,0], "ISLES2015":[1,3,5,2], "TBI":[1,3,5,2], "WMH":[1,3],"camcan":[3,5]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG_TBI_WMH/UNET/UNET_BRATS_ATLAS_MSSEG_TBI_WMH_RAND_BEST_BRATS.pth", "UNetv2", 6, {"BRATS":[1,3,4,5], "ATLAS":[3], "MSSEG":[1,3,4,5,0], "ISLES2015":[1,3,5,2], "TBI":[1,3,5,2], "WMH":[1,3],'camcan':[3]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG_TBI_WMH/UNET/UNET_BRATS_ATLAS_MSSEG_TBI_WMH_RAND_BEST_ATLAS.pth", "UNet", 6, {"BRATS":[1,3,4,5], "ATLAS":[3], "MSSEG":[1,3,4,5,0], "ISLES2015":[1,3,5,2], "TBI":[1,3,5,2], "WMH":[1,3]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG_TBI_WMH/UNET/UNET_BRATS_ATLAS_MSSEG_TBI_WMH_RAND_BEST_MSSEG.pth", "UNetv2", 6, {"BRATS":[1,3,4,5], "ATLAS":[3], "MSSEG":[1,3,4,5,0], "ISLES2015":[1,3,5,2], "TBI":[1,3,5,2], "WMH":[1,3]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG_TBI_WMH/UNET/UNET_BRATS_ATLAS_MSSEG_TBI_WMH_RAND_BEST_TBI.pth", "UNet", 6, {"BRATS":[1,3,4,5], "ATLAS":[3], "MSSEG":[1,3,4,5,0], "ISLES2015":[1,3,5,2], "TBI":[1,3,5,2], "WMH":[1,3]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG_TBI_WMH/UNET/UNET_BRATS_ATLAS_MSSEG_TBI_WMH_RAND_BEST_WMH.pth", "UNet", 6, {"BRATS":[1,3,4,5], "ATLAS":[3], "MSSEG":[1,3,4,5,0], "ISLES2015":[1,3,5,2], "TBI":[1,3,5,2], "WMH":[1,3]}),

    
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG_TBI_WMH/MSFN/MSFN_BRATS_ATLAS_MSSEG_TBI_WMH_RAND_Epoch_199.pth", "MSFN", 6, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG_TBI_WMH/MSFN/MSFN_BRATS_ATLAS_MSSEG_TBI_WMH_RAND_redo_2_Epoch_199.pth", "MSFN", 6, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG_TBI_WMH/MSFN/MSFN_BRATS_ATLAS_MSSEG_TBI_WMH_RAND_redo_2_BEST_BRATS.pth", "MSFN", 6, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG_TBI_WMH/MSFN/MSFN_BRATS_ATLAS_MSSEG_TBI_WMH_RAND_redo_2_BEST_ATLAS.pth", "MSFN", 6, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG_TBI_WMH/MSFN/MSFN_BRATS_ATLAS_MSSEG_TBI_WMH_RAND_redo_2_BEST_TBI.pth", "MSFN", 6, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG_TBI_WMH/MSFN/MSFN_BRATS_ATLAS_MSSEG_TBI_WMH_RAND_redo_2_BEST_WMH.pth", "MSFN", 6, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),

    # BRATS ATLAS MSSEG TBI
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG_TBI/UNET/UNET_BRATS_ATLAS_MSSEG_TBI_Epoch_199.pth", "UNet", 6, {"BRATS":[1,3,4,5], "ATLAS":[3], "MSSEG":[1,3,4,5,0], "ISLES2015":[1,3,5,2], "TBI":[1,3,5,2], "WMH":[1,3]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG_TBI/UNET/UNET_BRATS_ATLAS_MSSEG_TBI_BEST_BRATS.pth", "UNet", 6, {"BRATS":[1,3,4,5], "ATLAS":[3], "MSSEG":[1,3,4,5,0], "ISLES2015":[1,3,5,2], "TBI":[1,3,5,2], "WMH":[1,3]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG_TBI/UNET/UNET_BRATS_ATLAS_MSSEG_TBI_BEST_ATLAS.pth", "UNet", 6, {"BRATS":[1,3,4,5], "ATLAS":[3], "MSSEG":[1,3,4,5,0], "ISLES2015":[1,3,5,2], "TBI":[1,3,5,2], "WMH":[1,3]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG_TBI/UNET/UNET_BRATS_ATLAS_MSSEG_TBI_BEST_MSSEG.pth", "UNet", 6, {"BRATS":[1,3,4,5], "ATLAS":[3], "MSSEG":[1,3,4,5,0], "ISLES2015":[1,3,5,2], "TBI":[1,3,5,2], "WMH":[1,3]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG_TBI/UNET/UNET_BRATS_ATLAS_MSSEG_TBI_BEST_TBI.pth", "UNet", 6, {"BRATS":[1,3,4,5], "ATLAS":[3], "MSSEG":[1,3,4,5,0], "ISLES2015":[1,3,5,2], "TBI":[1,3,5,2], "WMH":[1,3]}),



    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS/MSFN/MSFN_BRATS_RAND_BEST_BRATS.pth", "MSFN", 4, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS/MSFN_PAIRED/RAND/MSFN_PAIRED_BRATS_RAND_BEST_BRATS.pth", "MSFNP", 4, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS/MSFN/MSFN_BRATS_RAND_BEST_BRATS.pth", "MSFN", 4, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS/UNET/RAND/UNET_BRATS_RAND_BEST_BRATS.pth", "UNet", 4, {"BRATS":[0,1,2,3], "ATLAS":[1], "MSSEG":[0,1,2,3], "ISLES2015":[0,1,3,2], "TBI":[0,1,3,2], "WMH":[0,1]}),

    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS/MSFN/redo/MSFN_BRATS_RAND_redo_BEST_BRATS.pth", "MSFN", 4, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS/UNET/ALL/UNET_BRATS_ALL_BEST_BRATS.pth", "UNet", 4, {"BRATS":[0,1,2,3], "ATLAS":[1], "MSSEG":[], "ISLES2015":[0,1,3,2]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS/UNET/RAND/UNET_BRATS_RAND_BEST_BRATS.pth", "UNet", 4, {"BRATS":[0,1,2,3], "ATLAS":[1], "MSSEG":[], "ISLES2015":[0,1,3,2]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS/MSFN_PAIRED/ALL/MSFN_PAIRED_BRATS_ALL_BEST_BRATS.pth", "MSFNP", 4, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/MSFN_BRATS_ALL_BEST_BRATS.pth", "MSFN", 4, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/HEM_BRATS_ALL_BEST_BRATS.pth", "HEM spatial attention", 4, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),


    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ATLAS/UNET/UNET_ATLAS_ALL_BATCH_3_BEST_ATLAS.pth", "UNet", 1, {"BRATS":[], "ATLAS":[1], "MSSEG":[], "ISLES2015":[0,1,3,2]}),

    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ATLAS_MSSEG_TBI_WMH/UNET/UNET_ATLAS_MSSEG_TBI_WMH_Epoch_199.pth", "UNet", 6, {"BRATS":[1,3,4,5], "ATLAS":[3], "MSSEG":[1,3,4,5,0], "ISLES2015":[1,3,5,2], "TBI":[1,3,5,2], "WMH":[1,3]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ATLAS_MSSEG_TBI_WMH/UNET/UNET_ATLAS_MSSEG_TBI_WMH_BEST_ATLAS.pth", "UNet", 6, {"BRATS":[1,3,4,5], "ATLAS":[3], "MSSEG":[1,3,4,5,0], "ISLES2015":[1,3,5,2], "TBI":[1,3,5,2], "WMH":[1,3]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ATLAS_MSSEG_TBI_WMH/UNET/UNET_ATLAS_MSSEG_TBI_WMH_BEST_TBI.pth", "UNet", 6, {"BRATS":[1,3,4,5], "ATLAS":[3], "MSSEG":[1,3,4,5,0], "ISLES2015":[1,3,5,2], "TBI":[1,3,5,2], "WMH":[1,3]}),

    # Pre train nets
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_TBI/Pre_train/MSFN/MSFN_TBI_ALL_FIRST_50_PRE_TRAIN_BRATS_ATLAS_MSSEG_Epoch_299.pth", "MSFN", 6, {"BRATS":[1,3,4,5], "ATLAS":[3], "MSSEG":[1,3,4,5,0], "ISLES2015":[1,3,5,2], "TBI":[1,3,5,2], "WMH":[1,3]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_TBI/MSFN/MSFN_TBI_ALL_FIRST_50_BEST_TBI.pth", "MSFN", 6, {"BRATS":[1,3,4,5], "ATLAS":[3], "MSSEG":[1,3,4,5,0], "ISLES2015":[1,3,5,2], "TBI":[1,3,5,2], "WMH":[1,3]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_TBI/Pre_train/UNET/UNET_TBI_ALL_FIRST_50_PRE_TRAIN_BRATS_ATLAS_MSSEG_Epoch_249.pth", "UNet", 5, {"BRATS":[1,2,3,4], "ATLAS":[2], "MSSEG":[1,2,3,4,0], "ISLES2015":[1,2,4,3], "TBI":[1,2,4,3], "WMH":[1,2]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_TBI/UNET/UNET_TBI_ALL_FIRST_50_BEST_TBI.pth", "UNet", 4, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[0,2,3,1], "WMH":[]}),


    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_TBI/UNET/UNET_TBI_ALL_BEST_TBI.pth", "UNet", 4, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[0,2,3,1], "WMH":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_TBI/Pre_train/UNET/UNET_TBI_ALL_PRE_TRAINED_BRATS_ATLAS_MSSEG_BEST_TBI.pth", "UNet", 5, {"BRATS":[1,2,3,4], "ATLAS":[2], "MSSEG":[1,2,3,4,0], "ISLES2015":[1,2,4,3], "TBI":[1,2,4,3], "WMH":[1,2]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_TBI/MSFN/MSFN_TBI_ALL_BEST_TBI.pth", "MSFN", 5, {"BRATS":[1,2,3,4], "ATLAS":[2], "MSSEG":[1,2,3,4,0], "ISLES2015":[1,2,4,3], "TBI":[1,2,4,3], "WMH":[1,2]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_TBI/Pre_train/MSFN/MSFN_TBI_ALL_PRE_TRAINED_BRATS_ATLAS_MSSEG_BEST_TBI.pth", "MSFN", 5, {"BRATS":[1,2,3,4], "ATLAS":[2], "MSSEG":[1,2,3,4,0], "ISLES2015":[1,2,4,3], "TBI":[1,2,4,3], "WMH":[1,2]}),


    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ATLAS_MSSEG_TBI_WMH/MSFN/MSFN_ATLAS_MSSEG_TBI_WMH_BEST_ATLAS.pth", "MSFN", 5, {"BRATS":[1,2,3,4], "ATLAS":[2], "MSSEG":[1,2,3,4,0], "ISLES2015":[1,2,4,3], "TBI":[1,2,4,3], "WMH":[1,2]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ATLAS_MSSEG_TBI_WMH/MSFN/MSFN_ATLAS_MSSEG_TBI_WMH_Epoch_149.pth", "MSFN", 5, {"BRATS":[1,2,3,4], "ATLAS":[2], "MSSEG":[1,2,3,4,0], "ISLES2015":[1,2,4,3], "TBI":[1,2,4,3], "WMH":[1,2]}),


    # ///////////////////////////////// START OF MESSY SINGLE TRAIN FILES  //////////////////////////////////////////////
    #  TRAIN JUST ATLAS
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ATLAS/UNET/UNET_ATLAS_ALL_BATCH_3_BEST_ATLAS.pth", "UNet", 1, {"BRATS":[], "ATLAS":[0], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    # bad msfn
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ATLAS/MSFN/MSFN_ATLAS_ALL_Epoch_199.pth", "MSFN", 1, {"BRATS":[], "ATLAS":[0], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),

    # TRAIN JUST BRATS

    # TRAIN JUST MSSEG
    # not theory unets
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_MSSEG/UNET/RAND/UNET_MSSEG_RAND_BEST_MSSEG.pth", "UNetv2", 5, {"BRATS":[1,2,3,4], "ATLAS":[2], "MSSEG":[1,2,3,4,0], "ISLES2015":[1,2,4,3], "TBI":[1,2,4,0], "WMH":[1,2]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_MSSEG/UNET/UNET_MSSEG_ALL_BATCH_2_BEST_MSSEG.pth", "UNet", 5, {"BRATS":[1,2,3,4], "ATLAS":[2], "MSSEG":[1,2,3,4,0], "ISLES2015":[1,2,4,3], "TBI":[1,2,4,0], "WMH":[1,2]}),
    # bad msfn
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_MSSEG/MSFN/MSFN_MSSEG_ALL_BATCH_2_BEST_MSSEG.pth", "MSFN", 1, {"BRATS":[], "ATLAS":[0], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    # 2x channels in fusion
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_MSSEG/MSFNP/MSFNP_MSSEG_ALL_BATCH_2_BEST_MSSEG.pth", "MSFNP", 1, {"BRATS":[], "ATLAS":[0], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    # bad ms fusion
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_MSSEG/HEM_SPATIAL_ATTENTION/ALL/HEM_MSSEG_ALL_BATCH_2_BEST_MSSEG.pth", "HEM spatial attention", 1, {"BRATS":[], "ATLAS":[0], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_MSSEG/HEM_SPATIAL_ATTENTION/RAND/HEM_Spatial_Attention_MSSEG_RAND_BEST_MSSEG.pth", "HEM spatial attention", 1, {"BRATS":[], "ATLAS":[0], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),

    # TRAIN JUST ISLES
    # not theory unet
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ISLES/UNET/All/UNET_ISLES_ALL_BATCH_2_Epoch_199.pth", "UNet", 4, {"BRATS":[1,2,0,3], "ATLAS":[2], "MSSEG":[1,2,3,0], "ISLES2015":[1,2,3,0], "TBI":[1,2,3,0], "WMH":[1,2]}),
    # bad msfn
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ISLES/MSFN/MSFN_ISLES_RAND_BEST_ISLES.pth", "MSFN", 1, {"BRATS":[], "ATLAS":[0], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    #  double number of channels
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ISLES/MSFNP/MSFNP_NO_SHUFFLE_ISLES_ALL_BEST_ISLES.pth", "MSFNP", 1, {"BRATS":[], "ATLAS":[0], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    # normal number of channels
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ISLES/MSFNP/MSFNP_ISLES_RAND_BEST_ISLES.pth", "MSFNP", 1, {"BRATS":[], "ATLAS":[0], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    # good ms fusion
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ISLES/HEM_SPATIAL_ATTENTION/HEM_ISLES_RAND_BEST_ISLES.pth", "HEM spatial attention", 1, {"BRATS":[], "ATLAS":[0], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    # bad ms fusion
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ISLES/HEM_SPATIAL_ATTENTION/HEM_ISLES_ALL_BATCH_2_BEST_ISLES.pth", "HEM spatial attention", 1, {"BRATS":[], "ATLAS":[0], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),


    # TRAIN JUST TBI
    # theory unet
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_TBI/UNET/UNET_TBI_ALL_BEST_TBI.pth", "UNet", 4, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[0,2,3,1], "WMH":[]}),
    # bad msfn
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_TBI/MSFN/MSFN_TBI_ALL_BEST_TBI.pth", "MSFN", 4, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    # 2x chans
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_TBI/MSFNP/MSFNP_TBI_ALL_BEST_TBI.pth", "MSFNP", 4, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    # bad ms fusion
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_TBI/HEM/HEM_TBI_ALL_BEST_TBI.pth", "HEM spatial attention", 4, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_TBI/HEM/HEM_TBI_RAND_BEST_TBI.pth", "HEM spatial attention", 4, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),

    
    # TRAIN JUST WMH
    # unet v2
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_WMH/UNET/UNET_WMH_ALL_BEST_WMH.pth", "UNet", 2, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[0,1]}),
    # bad hem
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_WMH/HEM_WMH_ALL_BEST_WMH.pth", "HEM spatial attention", 2, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[0,1]}),
    # bad msfn
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_WMH/MSFN_WMH_ALL_BEST_WMH.pth", "MSFN", 2, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[0,1]}),
    



    # //////////////////////////////////// CLEAN SINGLE TRAIN FILES ////////////////////////////////////////////
    # train brats
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/HEM_BRATS_RAND_BEST_BRATS.pth", "HEM spatial attention", 2, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    # train wmh
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/HEM_WMH_RAND_BEST_WMH.pth", "HEM spatial attention", 2, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/MSFNP_WMH_ALL_BEST_WMH.pth", "MSFNP", 2, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/MSFNP_WMH_RAND_BEST_WMH.pth", "MSFNP", 2, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/UNET_WMH_RAND_BEST_WMH.pth", "UNet", 2, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[0,1]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/MSFN_WMH_RAND_BEST_WMH.pth", "MSFN", 2, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    
    
    
    
    # # train atlas
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/MSFNP_ATLAS_ALL_BEST_ATLAS.pth", "MSFNP", 2, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/MSFN_ATLAS_ALL_BEST_ATLAS.pth", "MSFN", 2, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    # # train msseg
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/MSFNP_MSSEG_rand_BEST_MSSEG.pth", "MSFNP", 2, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/MSFN_MSSEG_rand_BEST_MSSEG.pth", "MSFN", 2, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    # # train tbi
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/MSFNP_TBI_rand_BEST_TBI.pth", "MSFNP", 2, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/MSFN_TBI_rand_BEST_TBI.pth", "MSFN", 2, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/UNET_TBI_rand_BEST_TBI.pth", "UNet", 4, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[0,2,3,1], "WMH":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_TBI/HEM/HEM_TBI_RAND_BEST_TBI.pth", "HEM spatial attention", 4, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[0,2,3,1], "WMH":[]}),    
    # # train isles
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/MSFN_ISLES_ALL_BEST_ISLES.pth", "MSFN", 2, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[], "TBI":[], "WMH":[]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/UNET_ISLES_rand_BEST_ISLES.pth", "UNet", 4, {"BRATS":[], "ATLAS":[], "MSSEG":[], "ISLES2015":[1,2,3,0], "TBI":[], "WMH":[]}),
    
    # train brats atlas msseg wmh
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/UNET_BRATS_ATLAS_MSSEG_WMH_BEST_BRATS.pth", "UNet", 5, {"BRATS":[1,2,3,4], "ATLAS":[2], "MSSEG":[1,2,3,4,0], "ISLES2015":[1,2,4,0], "TBI":[1,2,4,3], "WMH":[1,2]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/UNET_BRATS_ATLAS_MSSEG_WMH_BEST_MSSEG.pth", "UNet", 5, {"BRATS":[1,2,3,4], "ATLAS":[2], "MSSEG":[1,2,3,4,0], "ISLES2015":[1,2,4,0], "TBI":[1,2,4,3], "WMH":[1,2]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/UNET_BRATS_ATLAS_MSSEG_WMH_BEST_WMH.pth", "UNet", 5, {"BRATS":[1,2,3,4], "ATLAS":[2], "MSSEG":[1,2,3,4,0], "ISLES2015":[1,2,4,0], "TBI":[1,2,4,3], "WMH":[1,2],"camcan":[2]}),
    
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/MSFN_BRATS_ATLAS_MSSEG_WMH_BEST_BRATS.pth", "MSFN", 5, {"BRATS":[0,2,3,4], "ATLAS":[2], "MSSEG":[0,2,3,4,1], "ISLES2015":[0,2,4,3], "TBI":[0,2,4,1], "WMH":[0,2]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/MSFN_BRATS_ATLAS_MSSEG_WMH_BEST_ATLAS.pth", "MSFN", 5, {"BRATS":[0,2,3,4], "ATLAS":[2], "MSSEG":[0,2,3,4,1], "ISLES2015":[0,2,4,3], "TBI":[0,2,4,1], "WMH":[0,2]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/MSFN_BRATS_ATLAS_MSSEG_WMH_BEST_WMH.pth", "MSFN", 5, {"BRATS":[0,2,3,4], "ATLAS":[2], "MSSEG":[0,2,3,4,1], "ISLES2015":[0,2,4,3], "TBI":[0,2,4,1], "WMH":[0,2]}),

    # train brats atlas tbi wmh
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/UNET_BRATS_ATLAS_TBI_WMH_BEST_ATLAS.pth", "UNet", 5, {"BRATS":[0,2,3,4], "ATLAS":[2], "MSSEG":[0,2,3,4,1], "ISLES2015":[0,2,4,3], "TBI":[0,2,4,1], "WMH":[0,2], "camcan":[2,4]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/UNET_BRATS_ATLAS_TBI_WMH_BEST_BRATS.pth", "UNet", 5, {"BRATS":[0,2,3,4], "ATLAS":[2], "MSSEG":[0,2,3,4,1], "ISLES2015":[0,2,4,3], "TBI":[0,2,4,1], "WMH":[0,2]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/UNET_BRATS_ATLAS_TBI_WMH_BEST_TBI.pth", "UNet", 5, {"BRATS":[0,2,3,4], "ATLAS":[2], "MSSEG":[0,2,3,4,1], "ISLES2015":[0,2,4,3], "TBI":[0,2,4,1], "WMH":[0,2]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/UNET_BRATS_ATLAS_TBI_WMH_BEST_WMH.pth", "UNet", 5, {"BRATS":[0,2,3,4], "ATLAS":[2], "MSSEG":[0,2,3,4,1], "ISLES2015":[0,2,4,3], "TBI":[0,2,4,1], "WMH":[0,2]}),
    
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/MSFN_BRATS_ATLAS_TBI_WMH_BEST_BRATS.pth", "MSFN", 5, {"BRATS":[0,2,3,4], "ATLAS":[2], "MSSEG":[0,2,3,4,1], "ISLES2015":[0,2,4,3], "TBI":[0,2,4,1], "WMH":[0,2]}),
    
    
    # train brats msseg tbi wmh
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/MSFN_BRATS_MSSEG_TBI_WMH_BEST_BRATS.pth", "MSFN", 5, {"BRATS":[0,2,3,4], "ATLAS":[2], "MSSEG":[0,2,3,4,1], "ISLES2015":[0,2,4,3], "TBI":[0,2,4,1], "WMH":[0,2]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/MSFN_BRATS_MSSEG_TBI_WMH_BEST_MSSEG.pth", "MSFN", 5, {"BRATS":[0,2,3,4], "ATLAS":[2], "MSSEG":[0,2,3,4,1], "ISLES2015":[0,2,4,3], "TBI":[0,2,4,1], "WMH":[0,2]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/MSFN_BRATS_MSSEG_TBI_WMH_BEST_TBI.pth", "MSFN", 5, {"BRATS":[0,2,3,4], "ATLAS":[2], "MSSEG":[0,2,3,4,1], "ISLES2015":[0,2,4,3], "TBI":[0,2,4,1], "WMH":[0,2]}),
    Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/UNET_BRATS_MSSEG_TBI_WMH_BEST_BRATS.pth", "UNet", 6, {"BRATS":[1,3,4,5], "ATLAS":[3], "MSSEG":[1,3,4,5,0], "ISLES2015":[1,3,5,0], "TBI":[1,3,5,2], "WMH":[1,3],"camcan":[3,5]}),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/UNET_BRATS_MSSEG_TBI_WMH_BEST_MSSEG.pth", "UNet", 6, {"BRATS":[1,3,4,5], "ATLAS":[3], "MSSEG":[1,3,4,5,0], "ISLES2015":[1,3,5,0], "TBI":[1,3,5,2], "WMH":[1,3]}),


    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ATLAS/MSFN_ATLAS_20_Epoch_499.pth", "MSFN", 0, { "ATLAS" :[0]} ),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ATLAS/MSFN_ATLAS_20_PRE_TRAIN_BRATS_MSSEG_TBI_WMH_BEST_ATLAS.pth", "MSFN", 0, { "ATLAS" :[0]} ),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ATLAS/UNET_ATLAS_20_BEST_ATLAS.pth", "UNet", 1, { "ATLAS" :[0]} ),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ATLAS/UNET_ATLAS_20_PRE_TRAIN_BRATS_MSSEG_TBI_WMH_BEST_ATLAS.pth", "UNet", 6, { "ATLAS" :[3]} ),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS/MSFN_BRATS_FIRST_20_BEST_BRATS.pth", "MSFN", 0, { "BRATS" :[0, 1, 2, 3]} ),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS/UNET_BRATS_ALL_FIRST_20_BEST_BRATS.pth", "UNet", 4, { "BRATS" :[0, 1, 2, 3]} ),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS/UNET_BRATS_ALL_FIRST_20_PRE_TRAIN_ATLAS_MSSEG_TBI_WMH_BEST_BRATS.pth", "UNet", 6, { "BRATS" :[1,3,4,5]} ),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS/MSFN_BRATS_20_PRE_TRAIN_ATLAS_MSSEG_TBI_WMH_Epoch_949.pth", "MSFN", 6, { "BRATS" :[1,3,4,5]} ),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_MSSEG/MSFN_MSSEG_20_BEST_MSSEG.pth", "MSFN", 0, { "MSSEG " :[0, 2, 3, 4, 1]} ),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_MSSEG/MSFN_MSSEG_20_PRE_TRAIN_BRATS_ATLAS_TBI_WMH_BEST_MSSEG.pth", "MSFN", 0, { "MSSEG" :[0, 2, 3, 4, 1]} ),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_MSSEG/UNET_MSSEG_20_BEST_MSSEG.pth", "UNet", 5, { "MSSEG" :[1, 2, 3, 4, 0]} ),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_MSSEG/UNET_MSSEG_20_PRE_TRAIN_BRATS_ATLAS_TBI_WMH_BEST_MSSEG.pth", "UNet", 5, { "MSSEG" :[0, 2, 3, 4, 1]} ),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_TBI/MSFN/MSFN_TBI_ALL_FIRST_50_BEST_TBI.pth", "MSFN", 0, { "TBI" :[0, 2, 3, 1]} ),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_TBI/MSFN_TBI_50_PRE_TRAIN_BRATS_ATLAS_MSSEG_WMH_BEST_TBI.pth", "MSFN", 0, { "TBI" :[0, 2, 3, 1]} ),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_TBI/UNET/UNET_TBI_ALL_FIRST_50_BEST_TBI.pth", "UNet", 4, { "TBI" :[0, 2, 3, 1]} ),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_TBI/UNET_TBI_50_PRE_TRAIN_BRATS_ATLAS_MSSEG_WMH_BEST_TBI.pth", "UNet", 5, { "TBI" :[1,2,4,3]} ),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_WMH/MSFN_WMH_FIRST_20_BEST_WMH.pth", "MSFN", 0, { "WMH" :[0, 1]} ),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_WMH/UNET_WMH_FIRST_20_BEST_WMH.pth", "UNet", 2, { "WMH" :[0, 1]} ),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_WMH/UNET_WMH_FIRST_20_PRE_TRAIN_BRATS_ATLAS_MSSEG_TBI_BEST_WMH.pth", "UNet", 6, { "WMH" :[1,3]} ),
    

    # TRAIN ALL
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES_MSSEG_TBI_WMH/UNET_BRATS_ATLAS_ISLES_MSSEG_TBI_WMH_BEST_BRATS.pth", "UNet", 7, {"BRATS":[2,4,5,6], "ATLAS":[4], "MSSEG":[2,4,5,6,0], "ISLES2015":[2,4,6,1], "TBI":[2,4,6,3], "WMH":[2,4]} ),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES_MSSEG_TBI_WMH/UNET_BRATS_ATLAS_ISLES_MSSEG_TBI_WMH_BEST_TBI.pth", "UNet", 7, {"BRATS":[2,4,5,6], "ATLAS":[4], "MSSEG":[2,4,5,6,0], "ISLES2015":[2,4,6,1], "TBI":[2,4,6,3], "WMH":[2,4]} ),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES_MSSEG_TBI_WMH/UNET_BRATS_ATLAS_ISLES_MSSEG_TBI_WMH_BEST_WMH.pth", "UNet", 7, {"BRATS":[2,4,5,6], "ATLAS":[4], "MSSEG":[2,4,5,6,0], "ISLES2015":[2,4,6,1], "TBI":[2,4,6,3], "WMH":[2,4]} ),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES_MSSEG_TBI_WMH/UNET_BRATS_ATLAS_ISLES_MSSEG_TBI_WMH_Epoch_199.pth", "UNet", 7, {"BRATS":[2,4,5,6], "ATLAS":[4], "MSSEG":[2,4,5,6,0], "ISLES2015":[2,4,6,1], "TBI":[2,4,6,3], "WMH":[2,4],"camcan":[4,6]} ),

    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES_MSSEG_TBI_WMH/MSFN_BRATS_ATLAS_ISLES_MSSEG_TBI_WMH_BEST_BRATS.pth", "MSFN", 0, { "WMH" :[0, 1]} ),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES_MSSEG_TBI_WMH/MSFN_BRATS_ATLAS_ISLES_MSSEG_TBI_WMH_BEST_ATLAS.pth", "MSFN", 0, { "WMH" :[0, 1]} ),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES_MSSEG_TBI_WMH/MSFN_BRATS_ATLAS_ISLES_MSSEG_TBI_WMH_BEST_ISLES.pth", "MSFN", 0, { "WMH" :[0, 1]} ),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES_MSSEG_TBI_WMH/MSFN_BRATS_ATLAS_ISLES_MSSEG_TBI_WMH_BEST_MSSEG.pth", "MSFN", 0, { "WMH" :[0, 1]} ),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES_MSSEG_TBI_WMH/MSFN_BRATS_ATLAS_ISLES_MSSEG_TBI_WMH_BEST_TBI.pth", "MSFN", 0, { "WMH" :[0, 1]} ),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_ISLES_MSSEG_TBI_WMH/MSFN_BRATS_ATLAS_ISLES_MSSEG_TBI_WMH_BEST_WMH.pth", "MSFN", 0, { "WMH" :[0, 1]} ),

    # TRAIN BRATS ATLAS MSSEG TBI
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/MSFN_BRATS_ATLAS_MSSEG_TBI_BEST_ATLAS.pth", "MSFN", 0, { "WMH" :[0, 1]} ),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/MSFN_BRATS_ATLAS_MSSEG_TBI_BEST_BRATS.pth", "MSFN", 0, { "WMH" :[0, 1]} ),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/MSFN_BRATS_ATLAS_MSSEG_TBI_BEST_MSSEG.pth", "MSFN", 0, { "WMH" :[0, 1]} ),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/MSFN_BRATS_ATLAS_MSSEG_TBI_BEST_TBI.pth", "MSFN", 0, { "WMH" :[0, 1]} ),
    # Net("/home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/MSFN_BRATS_ATLAS_MSSEG_TBI_Epoch_199.pth", "MSFN", 0, { "WMH" :[0, 1]} ),


    ]

    device_id = 1
    cuda_id = "cuda:" + str(device_id)
    device = torch.device(cuda_id)
    torch.cuda.set_device(cuda_id)

    set_determinism(seed=0)
    print("determinism seed = 0")

    datasets = ["BRATS", "ATLAS", "MSSEG", "ISLES2015", "TBI", "WMH"]
    #REMOVE THIS
    datasets = ["ATLAS"]

    dataset_modalities = [[0,1,2,3],[0],[0,1,2,3,4],[0,1,2,3],[0,1,2,3],[0,1]]
    # REMOVE THIS
    dataset_modalities = [[0]]

    results = {}
    for net in nets:
        print("*************** TESTING NET " + str(net.file_path) + " **************")

        if net.ensemble == False:
            model = utils.create_net(net, device, cuda_id)
        else:
            net.init_nets(device,cuda_id)
            model = net

        for i, dataset in enumerate(datasets):
            print("************** TESTING DATSET " + dataset + " ***************")
            dataloader = utils.create_dataset(dataset)
            modalities = utils.create_modality_combinations(dataset_modalities[i])
            # REMOVE THIS
            modalities = [(0,)]
            # modalities = dataset_modalities[i],

            for combination in modalities:
                # print(combination)
                dsc_scores, seg_pix_vols, gt_pix_vols, detected, undetected, det_seg_sizes, undet_seg_sizes = test.test(model,
                    dataloader,
                    dataset,
                    combination,
                    net,
                    device,
                    dataset_modalities[i],
                    save_outputs=True,
                    save_path="/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TEST_ATLAS/UNET/",
                    detect_blobs = False)
                    # save_path="/home/sedm6251/projectMaterial/baseline_models/Combined_Training/TEST_MSSEG/UNET_All_modalities/UNET_All_modalities_")
 


