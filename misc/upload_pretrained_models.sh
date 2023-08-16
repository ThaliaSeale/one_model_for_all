#!/bin/bash
transfer_model() {
	scp $1 "wolf6273@htc-login.arc.ox.ac.uk:/home/wolf6273/one_model_for_all/trained_models"
}

# ISLES
transfer_model /home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG_TBI/UNET/UNET_BRATS_ATLAS_MSSEG_TBI_Epoch_199.pth

# ATLAS
transfer_model /home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/UNET_BRATS_MSSEG_TBI_WMH_BEST_BRATS.pth

# BRATS
transfer_model /home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_ATLAS_MSSEG_TBI_WMH/UNET/UNET_ATLAS_MSSEG_TBI_WMH_BEST_TBI.pth

# MSSEG
transfer_model /home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/UNET_BRATS_ATLAS_TBI_WMH_BEST_ATLAS.pth

# TBI
transfer_model /home/sedm6251/projectMaterial/baseline_models/Combined_Training/from_cluster/UNET_BRATS_ATLAS_MSSEG_WMH_BEST_MSSEG.pth

#WMH
transfer_model /home/sedm6251/projectMaterial/baseline_models/Combined_Training/TRAIN_BRATS_ATLAS_MSSEG_TBI/UNET/UNET_BRATS_ATLAS_MSSEG_TBI_Epoch_199.pth
