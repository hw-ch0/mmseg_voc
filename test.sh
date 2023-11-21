# CONFIG=$1
# CHECKPOINT=$2
# GPUS=$3


# mmdet example
# bash tools/dist_test.sh \
# 		 projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_voc.py \
# 		 experiments_epochs6/epoch_1.pth \
# 		 4 --eval mAP

bash tools/dist_test.sh \
		 configs/maskformer/maskformer_r50-d32_8xb2-160k_ade20k-512x512.py \
		 checkpoint/maskformer_r50-d32_8xb2-160k_ade20k-512x512_20221030_182724-3a9cfe45.pth \
		 4
