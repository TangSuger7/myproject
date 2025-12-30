torchrun --standalone --nproc_per_node 8 \
train.py  --epochs 30 \
    --dist True