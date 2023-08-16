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

# run_experiment "ATLAS_pretrained_limited" train_progressive.py ATLAS 0 0 1 "[630]" 
# run_experiment "ATLAS_pretrained_all" train_progressive.py ATLAS 0 0 0 "[300]" 
run_experiment "ATLAS_scratch_all" train_progressive.py ATLAS 0 1 0 "[300]" 
run_experiment "ATLAS_scratch_limited" train_progressive.py ATLAS 0 1 1 "[630]"

current_time=$(date "+%Y.%m.%d-%H.%M.%S")

# mkdir results/"ATLAS_naive_pretrained_limited_${current_time}"
# python train_multiple.py 1 1000 "ATLAS_naive_pretrained_limited_${current_time}" ATLAS 0 1 "[800]" | tee results/"ATLAS_naive_pretrained_limited_${current_time}"/log.txt

# mkdir results/"ATLAS_naive_pretrained_all_${current_time}"
# python train_multiple.py 1 1000 "ATLAS_naive_pretrained_all_${current_time}" ATLAS 0 0 "[640]" | tee results/"ATLAS_naive_pretrained_all_${current_time}"/log.txt