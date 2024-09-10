# mkdir -p results/240826_mask2former_crop_nquery5_lr1e-4_sch-poly_ep160k

# bash tools/dist_train.sh \
#     configs/mask2former/mask2former_r50_8xb2-160k_voc12aug-512x512.py \
#     4 \
#     --work-dir results/240826_mask2former_crop_nquery5_lr1e-4_sch-poly_ep160k >> results/240826_mask2former_crop_nquery5_lr1e-4_sch-poly_ep160k/log.txt

mkdir -p results/240830_mask2former_crop_nquery25_lr1e-4_sch-poly_ep160k_extend1

bash tools/dist_train.sh \
    configs/mask2former/mask2former_r50_8xb2-160k_voc12aug-512x512_extend1.py \
    4 \
    --work-dir results/240830_mask2former_crop_nquery25_lr1e-4_sch-poly_ep160k_extend1 >> results/240830_mask2former_crop_nquery25_lr1e-4_sch-poly_ep160k_extend1/log.txt
