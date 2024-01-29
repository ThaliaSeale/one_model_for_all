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

# run_experiment "MSSEG_pretrained_limited" train_progressive.py MSSEG 0 0 1 250 
# run_experiment "MSSEG_pretrained_all" train_progressive.py MSSEG 0 0 0 300
# run_experiment "MSSEG_from_scratch_limited" train_progressive.py MSSEG 0 0 0 250 
# run_experiment "MSSEG_from_scratch_all" train_progressive.py MSSEG 0 1 0 550 

# current_time=$(date "+%Y.%m.%d-%H.%M.%S")

# mkdir results/"MSSEG_naive_pretrained_limited_${current_time}"
# python train_multiple.py 1 1000 "MSSEG_naive_pretrained_limited_${current_time}" MSSEG 0 1 "[800]" | tee results/"MSSEG_naive_pretrained_limited_${current_time}"/log.txt

# mkdir results/"MSSEG_naive_pretrained_all_${current_time}"
# python train_multiple.py 1 1000 "MSSEG_naive_pretrained_all_${current_time}" MSSEG 0 0 "[100]" | tee results/"MSSEG_naive_pretrained_all_${current_time}"/log.txt

current_time=$(date "+%Y.%m.%d-%H.%M.%S")
mkdir results/"MSSEG_progressive_limited_${current_time}"
python train_progressive.py 1 700 "MSSEG_progressive_limited_${current_time}" MSSEG 0 0 1 "[150]" | tee results/"MSSEG_progressive_limited_${current_time}"/log.txt