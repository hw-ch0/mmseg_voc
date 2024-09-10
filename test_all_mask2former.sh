#!/bin/bash

# Set the list of iterations
iters=(120000 125000 130000 135000 140000 145000 150000 155000 160000)

# Loop through each value in the list
for iter in "${iters[@]}"
do
    # Export the number of threads
    export OMP_NUM_THREADS='4'

    # Run the command with the current value
    bash tools/dist_test.sh \
        configs/mask2former/mask2former_r50_8xb2-160k_voc12aug-512x512.py \
        results/240826_mask2former_crop_nquery10_lr1e-4_sch-poly_ep160k/iter_$iter.pth \
        4 --tta >> results/240826_mask2former_crop_nquery10_lr1e-4_sch-poly_ep160k/log_test.txt
done
