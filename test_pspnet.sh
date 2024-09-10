export OMP_NUM_THREADS='4'

# bash tools/dist_test.sh \
# 		 configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-80k_pascal-context-480x480.py \
# 		 checkpoint/best_checkpoint.pth \
# 		 4 --tta

bash tools/dist_test.sh \
		 configs/pspnet/pspnet_r50-d8_4xb4-40k_voc12aug-512x512.py \
		 pspnet_r50-d8_512x512_40k_voc12aug.pth \
		 4 --tta