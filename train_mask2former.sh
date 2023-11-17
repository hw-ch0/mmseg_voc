
# mmdet example
# bash tools/dist_train.sh \
# projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_voc_ft.py \
# 4 experiments \
# --resume-from checkpoints/co_deformable_detr_r50_1x_coco.pth

bash tools/dist_train.sh \
    configs/mask2former/mask2former_r50_8xb2-160k_voc12aug-512x512.py \
    4 \
    --work-dir work_train_mask2former_160k