run_experiment () {
    mkdir results/$1
    python $2 0 1000 $1 $3 $4 $5 $6 | tee results/$1/log.txt
}
# $1 is experiment name
# $2 is script name
# $3 is dataset name
# $4 is random drop
# $5 is pretrain
# $6 is limited_data

run_experiment "BRATS_pretrained_limited" train_progressive.py BRATS 0 0 1
# run_experiment "ISLES_pretrained_limited" train_progressive.py ISLES 0 0 1
# run_experiment "ISLES_scratch_all" train_progressive.py ISLES 0 1 0
# run_experiment "ISLES_scratch_limited" train_progressive.py ISLES 0 1 1

# mkdir results/ISLES_2000_pretrained_all
# python train_multiple.py 1 2000 ISLES_2000_pretrained_all ISLES 0 | tee results/ISLES_2000_pretrained_all/log.txt
