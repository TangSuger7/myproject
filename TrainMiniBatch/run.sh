cp /home/notebook/code/personal/80410839/Uni/__init__.py /home/oppoer/.local/lib/python3.10/site-packages/torch/__init__.py

# cp /home/notebook/code/personal/80410839/Uni/LAB/pytorch_distributed.py /home/oppoer/.local/lib/python3.10/site-packages/optuna_integration/pytorch_distributed/pytorch_distributed.py 
# cp /home/notebook/code/personal/80410839/Uni/LAB/__init__.py /home/oppoer/.local/lib/python3.10/site-packages/torch/__init__.py

# python -m torch.distributed.launch \
#     --nproc_per_node=8 \
#     --master_addr="127.0.0.1" \
#     --master_port=12345 \
#     my_train.py \
#      --port 12345 2>&1 | tee /home/notebook/code/personal/80410839/Uni/LAB/ALL_TRY/out.log

/home/notebook/data/personal/80410839/Env/gs2/bin/python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=8 \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=127.0.0.1:12345 \
    my_train.py \
    --port 12345 2>&1 | tee out.log