destination_root="wolf6273@htc-login.arc.ox.ac.uk:/data/engs-mlmi1/wolf6273/"
# ISLES
# scp -r /home/sedm6251/projectMaterial/baseline_models/ISLES2015/Data/images wolf6273@htc-login.arc.ox.ac.uk:/data/engs-mlmi1/wolf6273/ISLES/images
# scp -r /home/sedm6251/projectMaterial/baseline_models/ISLES2015/Data/labels wolf6273@htc-login.arc.ox.ac.uk:/data/engs-mlmi1/wolf6273/ISLES/labels

# ATLAS
img_path_ATLAS="/home/sedm6251/projectMaterial/skullStripped/ATLAS/ATLAS/normed_images"
seg_path_ATLAS="/home/sedm6251/projectMaterial/skullStripped/ATLAS/ATLAS/trimmed_labels_ints"
# scp -r $img_path_ATLAS "${destination_root}ATLAS/normed_images" 
scp -r $seg_path_ATLAS "${destination_root}ATLAS/trimmed_labels_ints"

# MSSEG
img_path="/home/sedm6251/projectMaterial/datasets/MSSEG_2016/Normed"
seg_path="/home/sedm6251/projectMaterial/datasets/MSSEG_2016/Labels"
# scp -r $img_path "${destination_root}MSSEG/Normed"
# scp -r $seg_path "${destination_root}MSSEG/Labels"

# TBI
train_img_path="/home/sedm6251/projectMaterial/datasets/TBI/Train/Images"
train_seg_path_FLAIR="/home/sedm6251/projectMaterial/datasets/TBI/Train/Labels_FLAIR"
train_seg_path_SWI="/home/sedm6251/projectMaterial/datasets/TBI/Train/Labels_SWI"
train_seg_path_Merged="/home/sedm6251/projectMaterial/datasets/TBI/Train/Labels_Merged"

val_img_path="/home/sedm6251/projectMaterial/datasets/TBI/Test/Images"
val_seg_path_FLAIR="/home/sedm6251/projectMaterial/datasets/TBI/Test/Labels_FLAIR"
val_seg_path_SWI="/home/sedm6251/projectMaterial/datasets/TBI/Test/Labels_SWI"
val_seg_path_Merged="/home/sedm6251/projectMaterial/datasets/TBI/Test/Labels_Merged"

# scp -r $train_img_path "${destination_root}TBI/Train/Images"
# scp -r $train_seg_path_Merged "${destination_root}TBI/Train/Labels_Merged"

# scp -r $val_img_path "${destination_root}TBI/Test/Images"
# scp -r $val_seg_path_Merged "${destination_root}TBI/Test/Labels_Merged"

# WMH 
img_path="/home/sedm6251/projectMaterial/datasets/WMH/Images"
seg_path="/home/sedm6251/projectMaterial/datasets/WMH/Segs"
# scp -r $img_path "${destination_root}WMH/Images"
# scp -r $seg_path "${destination_root}WMH/Segs"

# BRATS
img_path="/home/shared_space/data/BRATS_Decathlon_2016_17/BRATS_Normalised_with_brainmask/normed"
seg_path="/home/shared_space/data/BRATS_Decathlon_2016_17/BRATS_merged_labels_inc_edema"
# scp -r $img_path "${destination_root}BRATS/Images"
# scp -r $seg_path "${destination_root}BRATS/Labels"
