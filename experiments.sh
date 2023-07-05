# mkdir results/ISLES_limited_2000_iterations
# python train_progressive.py 0 2000 ISLES_limited_2000_iterations  ISLES 0 1 1 | tee results/ISLES_limited_2000_iterations/log.txt

# mkdir results/ISLES_2000_iterations
# python train_progressive.py 0 2000 ISLES_2000_iterations  ISLES 0 1 0 | tee results/ISLES_2000_iterations/log.txt

# mkdir ISLES_linear_combi_test2  
# python train_progressive.py 1 1000 ISLES_linear_combi_test2  ISLES 0 0 1 | tee results/ISLES_linear_combi_test2/log.txt

# experiment_name="ISLES_2000_iterations"
# mkdir results/$experiment_name
# python train_progressive.py 1 2000 $experiment_name  ISLES 0 1 0 | tee results/$experiment_name/log.txt

# experiment_name="ISLES_2000_pretrained"
# mkdir results/$experiment_name
# python train_multiple.py 1 2000 $experiment_name  ISLES 0 | tee results/$experiment_name/log.txt

# experiment_name="ISLES_linear_combi_test_2000"
# mkdir results/$experiment_name  
# python train_progressive

# experiment_name="ISLES_scratch_all"
# mkdir results/$experiment_name
# python train_progressive.py 0 2000 $experiment_name ISLES 0 1 0 | tee results/$experiment_name/log.txt

# experiment_name="ISLES_scratch_limited"
# mkdir results/$experiment_name
# python train_progressive.py 0 2000 $experiment_name ISLES 0 1 1 | tee results/$experiment_name/log.txt

experiment_name="ISLES_pretrained_all"
mkdir results/$experiment_name
python train_progressive.py 1 2000 $experiment_name ISLES 0 0 0 | tee results/$experiment_name/log.txt

experiment_name="ISLES_pretrained_limited"
mkdir results/$experiment_name
python train_progressive.py 1 2000 $experiment_name ISLES 0 0 1 | tee results/$experiment_name/log.txt

# experiment_name="ISLES_2000_pretrained_all"
# mkdir results/$experiment_name
# python train_multiple.py 0 2000 $experiment_name  ISLES 0 | tee results/$experiment_name/log.txt
