#!/bin/bash
# Run script
# Settings of training & test for different tasks.
# method="$1"
# task=$(python3 config.py --print_task)
# case "${task}" in
#     'DIS5K') epochs=500 && val_last=50 && step=5 ;;
#     'COD') epochs=150 && val_last=50 && step=5 ;;
#     'HRSOD') epochs=150 && val_last=50 && step=5 ;;
#     'General') epochs=200 && val_last=50 && step=5 ;;
#     'General-2K') epochs=250 && val_last=30 && step=2 ;;
#     'Matting') epochs=150 && val_last=50 && step=5 ;;
# esac

torchrun --standalone --nproc_per_node 1 \
finError.py  --epochs 20 \
    --dist True
    # --resume ${resume_weights_path} \