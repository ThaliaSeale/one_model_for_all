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


# run_experiment "TBI_reduced_pretrained_limited" train_progressive.py TBI 0 0 1 "[300,500]" 
# run_experiment "TBI_reduced_pretrained_all" train_progressive.py TBI 1 0 0 500 "[500,700]"
# run_experiment "TBI_scratch_limited" train_progressive.py TBI 0 1 1 500 "[250]"
# run_experiment "TBI_scratch_all" train_progressive.py TBI 0 1 0 500 "[500,620]"

# current_time=$(date "+%Y.%m.%d-%H.%M.%S")

# mkdir results/"TBI_naive_pretrained_limited_${current_time}"
# python train_multiple.py 1 1000 "TBI_naive_pretrained_limited_${current_time}" TBI 0 1 "[600,1200]" | tee results/"TBI_naive_pretrained_limited_${current_time}"/log.txt

# mkdir results/"TBI_naive_pretrained_all_${current_time}"
# python train_multiple.py 1 1000 "TBI_naive_pretrained_all_${current_time}" TBI 0 0 "[600,1200]" | tee results/"TBI_naive_pretrained_all_${current_time}"/log.txt

current_time=$(date "+%Y.%m.%d-%H.%M.%S")
mkdir results/"TBI_progressive_limited_${current_time}"
python train_progressive.py 0 700 "TBI_progressive_limited_${current_time}" TBI 0 0 1 "[100,200]" | tee results/"TBI_progressive_limited_${current_time}"/log.txt
