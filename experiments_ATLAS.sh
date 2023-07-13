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

run_experiment "ATLAS_pretrained_limited" train_progressive.py ATLAS 0 0 1 500
run_experiment "ATLAS_pretrained_all" train_progressive.py ATLAS 0 0 0 500
