#!/bin/bash

PY_ARGS=${@:1}

torchrun --nnodes=$MLP_WORKER_NUM --nproc_per_node=$MLP_WORKER_GPU --node_rank=$MLP_ROLE_INDEX \
--master_addr=$MLP_WORKER_0_HOST --master_port=$MLP_WORKER_0_PORT \
tools/train.py --dist_mode $PY_ARGS

#torchrun --nnodes=$NNODES --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=$master_ip:$master_port \
#tools/train.py --dist_mode $PY_ARGS


