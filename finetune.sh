
# mmdet example
# bash tools/dist_train.sh \
# projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_voc_ft.py \
# 4 experiments \
# --resume-from checkpoints/co_deformable_detr_r50_1x_coco.pth

bash tools/dist_train.sh \
configs/maskformer/maskformer_r50-d32_8xb2-160k_voc12-512x512_finetune.py \
4 \
--work-dir work_finetune_coco \
--resume
# --resume-from checkpoint/maskformer_r50-d32_8xb2-160k_ade20k-512x512_20221030_182724-3a9cfe45.pth