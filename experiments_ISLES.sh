run_experiment () {
    current_time=$(date "+%Y.%m.%d-%H.%M.%S")
    experiment_name="${1}_${current_time}"
    mkdir results/$experiment_name
    python $2 0 1000 $experiment_name $3 $4 $5 $6 $7 | tee results/$experiment_name/log.txt
}
# $1 is experiment name
# $2 is script name
# $3 is dataset name
# $4 is random drop
# $5 is pretrain
# $6 is limited_data

run_experiment "ISLES_pretrained_all" train_progressive.py ISLES 0 0 0 "[400,700,900]"
run_experiment "ISLES_pretrained_limited" train_progressive.py ISLES 0 0 1 "[450,600]" 
run_experiment "ISLES_scratch_all" train_progressive.py ISLES 0 1 0 "[400,700,900]" 
run_experiment "ISLES_scratch_limited" train_progressive.py ISLES 0 1 1 "[500,700]"

current_time=$(date "+%Y.%m.%d-%H.%M.%S")

mkdir results/"ISLES_naive_pretrained_limited_${current_time}"
python train_multiple.py 1 1000 "ISLES_naive_pretrained_limited_${current_time}" ISLES 0 1 "[500,700]" | tee results/"ISLES_naive_pretrained_limited_${current_time}"/log.txt

mkdir results/"ISLES_naive_pretrained_all_${current_time}"
python train_multiple.py 1 1000 "ISLES_naive_pretrained_all_${current_time}" ISLES 0 0 "[500,700]" | tee results/"ISLES_naive_pretrained_all_${current_time}"/log.txt