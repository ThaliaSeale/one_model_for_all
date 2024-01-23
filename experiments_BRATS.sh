run_experiment () {
    current_time=$(date "+%Y.%m.%d-%H.%M.%S")
    experiment_name="${1}_${current_time}"
    mkdir results/$experiment_name
    python $2 0 700 $experiment_name $3 $4 $5 $6 $7 | tee results/$experiment_name/log.txt
}

# run_experiment "BRATS_pretrained_limited" train_progressive.py BRATS 0 0 1
# run_experiment "BRATS_pretrained_all" train_progressive.py BRATS 0 0 0 "[400]" 
# run_experiment "BRATS_scratch_limited" train_progressive.py BRATS 0 1 1
# run_experiment "BRATS_scratch_all" train_progressive.py BRATS 0 1 0 "[150]" 

# mkdir results/BRATS_2000_pretrained_all
# python train_multiple.py 1 2000 BRATS_2000_pretrained_all BRATS 0 | tee results/BRATS_2000_pretrained_all/log.txt

current_time=$(date "+%Y.%m.%d-%H.%M.%S")

# mkdir results/"BRATS_naive_pretrained_limited_${current_time}"
# python train_multiple.py 1 700 "BRATS_naive_pretrained_limited_${current_time}" BRATS 0 1 "[350]" | tee results/"BRATS_naive_pretrained_limited_${current_time}"/log.txt

# mkdir results/"BRATS_naive_pretrained_all_${current_time}"
# python train_multiple.py 1 700 "BRATS_naive_pretrained_all_${current_time}" BRATS 0 0 "[1000]" | tee results/"BRATS_naive_pretrained_all_${current_time}"/log.txt

# mkdir results/"BRATS_naive_pretrained_all_${current_time}"
# # python train_multiple.py 1 700 "BRATS_naive_pretrained_all_${current_time}" BRATS 0 0 "[500]" 2 | tee results/"BRATS_naive_pretrained_all_${current_time}"/log.txt
# python train_multiple.py 1 700 "BRATS_naive_pretrained_all_${current_time}" BRATS 0 0 "[500]" 3 | tee results/"BRATS_naive_pretrained_all_${current_time}"/log.txt
# mkdir results/"BRATS_naive_pretrained_all_${current_time}"
# python train_multiple.py 1 700 "BRATS_naive_pretrained_all_${current_time}" BRATS 0 0 "[500]" 4 | tee results/"BRATS_naive_pretrained_all_${current_time}"/log.txt

# mkdir results/"BRATS_ATLAS_etc_naive_pretrained_all_${current_time}"
# python train_multiple.py 0 700 "BRATS_ATLAS_etc_naive_pretrained_all_${current_time}" BRATS_ATLAS_MSSEG_TBI_WMH 0 0 "[500]" 2 | tee results/"BRATS_ATLAS_etc_naive_pretrained_all_${current_time}"/log.txt
# mkdir results/"BRATS_ATLAS_etc_naive_pretrained_all_${current_time}"
# python train_multiple.py 0 700 "BRATS_ATLAS_etc_naive_pretrained_all_${current_time}" BRATS_ATLAS_MSSEG_TBI_WMH 0 0 "[500]" 3 | tee results/"BRATS_ATLAS_etc_naive_pretrained_all_${current_time}"/log.txt
# mkdir results/"BRATS_ATLAS_etc_naive_pretrained_all_${current_time}"
# python train_multiple.py 1 700 "BRATS_ATLAS_etc_naive_pretrained_all_${current_time}" BRATS_ATLAS_MSSEG_TBI_WMH 0 0 "[500]" 4 | tee results/"BRATS_ATLAS_etc_naive_pretrained_all_${current_time}"/log.txt

mkdir results/"BRATS_progressive_limited_${current_time}"
python train_progressive.py 0 700 "BRATS_progressive_limited_${current_time}" BRATS 0 0 1 "[200,500]" | tee results/"BRATS_progressive_limited_${current_time}"/log.txt