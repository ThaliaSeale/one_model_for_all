# One Model For All

Creating a CNN model architecture that can handle any input modalities for biomedical MRI segmentation. Currently networks are trained on datasets individually - e.g. on Brain Tumour separately to stroke. This ignores the possiblility that there may be potential to learn inherent features of "healthy" and "unhealthy" matter. The main problem with this is that MRIs come in multiple different modalities. Most network architectures require this channel dimensionality to be predefined at training. We would like to make use of modalities never seen at training and be able to take in an undefined number of modalities.

Networks are and required components are in the Nets folder, scripts used for pre-processing in pre_processing and all scripts for various train and test configurations are in the main folder.

# Scripts
## train_multiple.py
- Contains the code for training multiple databases with heterogeneous modalities
- Training Method is to have class balance across datasets
- Can train any of the nets in this project

### Arguments:
* 1 - Device ID -> the cuda GPU ID to use for training
* 2 - Epochs -> number of epochs to train
* 3 - Save Name -> What to save the model and tensorboard files as
* 4 - Dataset -> The dataset to train
* 5 - Randomly Drop (Bool) -> Whether or not to randomly drop modalities
* 6 - BRATS 2 Channel Segmentation -> whether different segmentation masks for brats should be used if e.g. FLAIR and T2 are dropped
* 7 - Learning Rate Lower Limit (If using reduce learning rate on plateau, this is lower limit)
* 8 - Model Type -> The type of model to train (U-Net, MSFN, ...)
* 9 - Augment Modalities (Bool) -> whether the data should be augmented by a non-linear combination of existing modaities

## test_master.py 
Contains all network file paths and the function calls to test networks - all arguments are within the script

## test.py 
Contains the actual testing code

## utils.py 
Various functions used across scripts

## blob detection.py 
Contains functions for connected component detection testing

## Net_to_test.py 
Class structures for testing networks

## create_modality.py 
functions for data augmentation pathway

# Folders:
pre_processing - all pre-processing functions used
Nets - all network architectures
HEMIS_Nets_Legacy - old experimental networks 
Experimental - any additional scripts for experimentation that did not get progressed further

# Packages and Versions Used
* MONAI version: 1.1.dev2241
* Numpy version: 1.23.3
* Pytorch version: 1.14.0.dev20221012+cu117
* Nibabel version: 4.0.2
* Pillow version: 7.0.0
* Tensorboard version: 2.10.1
* gdown version: 4.5.1
* TorchVision version: 0.15.0.dev20221012+cu117
* tqdm version: 4.64.1
* psutil version: 5.9.2
* pandas version: 1.5.1
