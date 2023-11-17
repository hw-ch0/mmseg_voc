
# # mmdet example
# # bash tools/dist_train.sh \
# # projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_voc_ft.py \
# # 4 experiments \
# # --resume-from checkpoints/co_deformable_detr_r50_1x_coco.pth

# bash tools/dist_train.sh \
#     configs/maskformer/maskformer_r50-d32_8xb2-20k_voc12aug-512x512_train.py \
#     4 \
#     --work-dir work_train_aug


# mmdet example
# bash tools/dist_train.sh \
# projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_voc_ft.py \
# 4 experiments \
# --resume-from checkpoints/co_deformable_detr_r50_1x_coco.pth

bash tools/dist_train.sh \
    configs/maskformer/maskformer_r50-d32_8xb2-20k_voc12aug-512x512_train.py \
    4 \
    --work-dir work_train_aug_320k \
