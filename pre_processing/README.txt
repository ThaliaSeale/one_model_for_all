

All databases
* Normalise using brainmask (function "normalise" found in image_normalisation.py, and copied/pasted in various individual scripts)

* Remove excess label (for when part of labelled image removed by skullstrip)
- remove_excess label.py



Skullstrip (used for ATLAS only I think)
- skull_strip_2.py



Merge labels (for databases with multiple ground truth labels)
BRATS - merge_labels.py
TBI - TBI_multi_channel_labels.py



For ATLAS and MSSEG I may have cropped - I can't remember if I did but easy to check if the dimensions of the files used in training are different from those in the raw data.

-> if they are cropped, the way this was done is in MSSEG_Normalise.py and crop_ATLAS.py respectively



Relevant scripts for each dataset:
BRATS - image_normalisation.py
ATLAS - image_normalisation.py (and potentially crop_ATLAS.py)
MSSEG - MSSEG_Normalise.py
WMH - WMH_pre_proc.py
TBI - pre_proc_TBI.py
CAMCAN - camcan_2_preproc.py


ISLES - data was already preprocessed on the machine so not processed further as advised at the time (files used were those ending "subtrmean div std v2")