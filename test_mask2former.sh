start_time=$(date +%s)

export OMP_NUM_THREADS='4'

# bash tools/dist_test.sh \
# 		 configs/mask2former/mask2former_r50_8xb2-160k_voc12aug-512x512.py \
# 		 results/240826_mask2former_crop_nquery5_lr1e-4_sch-poly_ep160k/best_mIoU_iter_160000.pth \
# 		 4 --tta


### best
bash tools/dist_test.sh \
		 configs/mask2former/mask2former_r50_8xb2-160k_voc12aug-512x512.py \
		 results/240826_mask2former_crop_nquery25_lr1e-4_sch-poly_ep160k/iter_150000.pth \
		 4 --tta

### extend1
# bash tools/dist_test.sh \
# 		 configs/mask2former/mask2former_r50_8xb2-160k_voc12aug-512x512_extend1.py \
# 		 results/240830_mask2former_crop_nquery25_lr1e-4_sch-poly_ep160k_extend1/iter_150000.pth \
# 		 4 --tta


end_time=$(date +%s)
elapsed_time=$((end_time - start_time))

# Print the elapsed time
echo "Elapsed time: $elapsed_time seconds"