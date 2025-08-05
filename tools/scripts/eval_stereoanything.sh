#!/bin/bash

source tools/scripts/constant.sh

dir_path="${CKPT_ROOT_DIR%/}/${1%/}/ckpt"

max_num=-1
for dir in "$dir_path"/epoch_*; do
    if [ -d "$dir" ]; then
        dir_name=$(basename "$dir")
        num_str=${dir_name#epoch_}
        num=$((10#$num_str))
        if [ $num -gt $max_num ]; then
            max_num=$num
        fi
    fi
done

pretrained_model="$dir_path/epoch_$max_num/pytorch_model.bin"
nproc_per_node=8
master_port=2335
cfg_file="cfgs/nmrf/nmrf_swint_sceneflow.py"

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node --master_port=$master_port \
tools/eval.py --dist_mode --cfg_file $cfg_file --pretrained_model $pretrained_model \
--eval_data_cfg_file cfgs/common/datasets/kitti12.py --eval_batch_size 1
echo "###########################################################"
echo

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node --master_port=$master_port \
tools/eval.py --dist_mode --cfg_file $cfg_file --pretrained_model $pretrained_model \
--eval_data_cfg_file cfgs/common/datasets/kitti15.py --eval_batch_size 1
echo "###########################################################"
echo

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node --master_port=$master_port \
tools/eval.py --dist_mode --cfg_file $cfg_file --pretrained_model $pretrained_model \
--eval_data_cfg_file cfgs/common/datasets/middlebury.py --eval_batch_size 1
echo "###########################################################"
echo

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node --master_port=$master_port \
tools/eval.py --dist_mode --cfg_file $cfg_file --pretrained_model $pretrained_model \
--eval_data_cfg_file cfgs/common/datasets/eth3d.py --eval_batch_size 1
echo "###########################################################"
echo
