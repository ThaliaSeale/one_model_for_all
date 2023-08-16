# PROGRESSIVE NETWORKS FOR TRANSFER LEARNING INMEDICAL IMAGE SEGMENTATION

Creating a CNN model architecture that can handle any input modalities for biomedical MRI segmentation. Currently networks are trained on datasets individually - e.g. on Brain Tumour separately to stroke. This ignores the possiblility that there may be potential to learn inherent features of "healthy" and "unhealthy" matter. The main problem with this is that MRI comes in multiple different modalities. Most network architectures require this channel dimensionality to be predefined at training. We would like to make use of modalities never seen at training and be able to take in an undefined number of modalities.
Based on base architecture from "One Model For All".

Machine learning (ML) has the potential for automating a wide variety of tasks in medical imaging, such as segmentation.
However, both the limited ability of ML models to generalise from the source domain to the target domain due to distribution shift and the simultaneous paucity of data from the target domain are obstacles to the deployment of ML in a clinical setting. 
A potential remedy for this issue is to use a model pre-trained on data from a related dataset or task that may later be *finetuned*.
This process adjusts model parameters to suit the target domain, using the limited data available. 
However, in some cases, pre-training actually negatively affects training, initialising the model parameters in a poor local minimum that causes the model to perform worse than training from scratch even after finetuning.

The code in this repository implements a Progressive U-Net, which uses a progressive network architecture to expand the capacity of a pre-trained U-Net to use any salient features obtained in pre-training while enabling the model to continue learning features unique to the new task.
The base model is a U-Net for image segmentation on 6 different 3D MRI image datasets which each focus on a different pathology and set of MRI image modalities, proposed by the "One Model for All" project.

# Folder descriptions

Networks and required components are in the Nets folder, scripts used for pre-processing in pre_processing and all scripts for various train and test configurations are in the main folder.

# Data
THE ORDER OF THE MODALITY CHANNELS IN EACH DATASET IS ASSUMED TO BE:
* BRATS: ["FLAIR", "T1", "T1c", "T2"]
* ATLAS: ["T1"]
* MSSEG: ["FLAIR","T1","T1c","T2", "DP"]
* ISLES: ["FLAIR", "T1", "T2", "DWI"]
* TBI: ["FLAIR", "T1", "T2", "SWI"]
* WMH: ["FLAIR", "T1"]

# Scripts

## `Nets/*_progressive_unet.py`

Progressive U-Nets for training on each of the datasets. There are individual files for each dataset as the forward hook code does not work outside of the file.

**The cuda device needs to be changed in these files as well as specified in using the command in the terminal.**

## `Nets/theory_UNET.py`

Contains code for the base U-Net and the progressive U-Net. `theory_UNET_progressive_cluster.py` also contains code for a progressive net with a reduced number of parameters.

## `misc`

Contains some shell scripts for downloading and uploading the data to a cluster.

## `experiments_*.sh`

Shell scripts for running experiments. 

## `train_progressive.py`

Contains code for training the progressive U-net. Based on the training procedure from `train_model.py` (see below).

### Arguments:

* 1 - Device ID -> the cuda GPU ID to use for training
* 2 - Epochs -> number of epochs to train
* 3 - Save Name -> What to save the model and tensorboard files as
* 4 - Dataset -> The dataset to train
* 5 - Randomly Drop (Bool) -> Whether or not to randomly drop modalities
* 6 - Pretrain -> Whether to train from scratch or use the progressive U-Net architecture with a pretrained column.
* 7 - Limited Data -> Whether to use the limited data size or to train on the full dataset.
* 8 - Epochs to drop learning rate -> List of epochs to drop the learning rate

## `train_multiple.py`
- Contains the code for training multiple databases with heterogeneous modalities
- Training Method is to have class balance across datasets
- Can train any of the nets in this project

### Arguments:
* 1 - Device ID -> the cuda GPU ID to use for training
* 2 - Epochs -> number of epochs to train
* 3 - Save Name -> What to save the model and tensorboard files as
* 4 - Dataset -> The dataset to train
* 5 - Randomly Drop (Bool) -> Whether or not to randomly drop modalities
* 6 - Pretrain -> Whether to train from scratch or use the progressive U-Net architecture with a pretrained column.

Example command:
python train_multiple.py 0 1000 my_save_name BRATS_ATLAS_ISLES 1
* means train on BraTS, ATLAS, and ISLES while randomly dropping modalities on GPU 0 for 1000 epochs and save a tensorboard log to my_save_name

## test_master.py 
### Arguments:
* 1 - Device ID -> the GPU ID to use for testing (e.g. 0 or 1)
* 2 - Dataset to Test -> the dataset to test (possible options are BRATS, ATLAS, MSSEG, ISLES, TBI, WMH)
* 3 - Modalities to test -> Which modalities to test on (e.g. 013 means test on modalities 0, 1 and 3
* 4 - Test all combinations -> If true, ignores the previous argument and instead tests all possible combinations of modalities for the given dataset

Contains all network file paths and calls to run the testing script. The file to test should be put in the list of Nets as an object of class Net which contains the filepath, network type, and channel mappings. Any other elements that you do not want to test that are in the Nets list should be commented out. Once a new net has been entered into the nets list, and its channel mappings entered it does not need to be changed for any other testing.

Example command:
python test_master.py 0 BRATS 012 0
* tests any uncommented network in the nets list on BRATS modalities 0,1,2 (FLAIR, T1, T1c) using GPU 0

## test.py 
Contains the actual testing code - prints out the overall DSC but has commented out lines that have further functionality

## utils.py 
Various functions used across scripts

## blob detection.py 
Contains functions for connected component detection testing

## Net_to_test.py 
Class structures for testing networks

## create_modality.py 
functions for data augmentation pathway

# Folders:
* pre_processing - all pre-processing functions used
* Nets - all network architectures
* HEMIS_Nets_Legacy - old experimental networks 
* Experimental - any additional scripts for experimentation that did not get progressed further

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
