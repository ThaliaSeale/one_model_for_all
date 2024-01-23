run_experiment () {
    mkdir results/$1
    python $2 0 1000 $1 $3 $4 $5 $6 $7 | tee results/$1/log.txt
}
# $1 is experiment name
# $2 is script name
# $3 is dataset name
# $4 is random drop
# $5 is pretrain
# $6 is limited_data

# run_experiment "WMH_pretrained_limited" train_progressive.py WMH 0 0 1 500
# run_experiment "WMH_pretrained_all" train_progressive.py WMH 0 0 0 "[600]" 
# run_experiment "WMH_from_scratch_limited" train_progressive.py WMH 0 1 1 500
# run_experiment "WMH_from_scratch_all" train_progressive.py WMH 0 1 0 "[700]" 

current_time=$(date "+%Y.%m.%d-%H.%M.%S")

# mkdir results/"WMH_naive_pretrained_limited_${current_time}"
# python train_multiple.py 1 1000 "WMH_naive_pretrained_limited_${current_time}" WMH 0 1 "[1000]" | tee results/"WMH_naive_pretrained_limited_${current_time}"/log.txt

# mkdir results/"WMH_naive_pretrained_all_${current_time}"
# python train_multiple.py 1 1000 "WMH_naive_pretrained_all_${current_time}" WMH 0 0 "[900]" | tee results/"WMH_naive_pretrained_all_${current_time}"/log.txt

mkdir results/"WMH_progressive_limited_${current_time}"
python train_progressive.py 0 700 "WMH_progressive_limited_${current_time}" WMH 0 0 1 "[600]" | tee results/"WMH_progressive_limited_${current_time}"/log.txt