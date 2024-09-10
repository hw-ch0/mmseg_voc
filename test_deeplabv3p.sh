export OMP_NUM_THREADS='4'

# bash tools/dist_test.sh \
# 		 configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-80k_pascal-context-480x480.py \
# 		 checkpoint/best_checkpoint.pth \
# 		 4 --tta

bash tools/dist_test.sh \
		 configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-40k_voc12aug-512x512.py \
		 results/work_train_deeplabv3plus_40k/iter_40000.pth \
		 4 --tta