results_dir="wolf6273@htc-login.arc.ox.ac.uk:/home/wolf6273/one_model_for_all/results/"
destination_dir="/home/thalia/one_model_for_all/results_tensorboard_hack"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
mkdir "ARC_experiments_${current_time}" 
scp -r $results_dir "${destination_dir}/ARC_experiments_${current_time}" 
