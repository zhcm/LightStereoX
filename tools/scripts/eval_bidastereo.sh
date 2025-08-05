#!/bin/bash
# set -x
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
master_port=2334
cfg_file="cfgs/bidastereo/sf_pretrain.py"


torchrun --nnodes=1 --nproc_per_node=1 --master_port=$master_port \
tools/eval.py --dist_mode --cfg_file $cfg_file --pretrained_model $pretrained_model \
--eval_data_cfg_file cfgs/common/sequence_datasets/sintel_clean.py --eval_batch_size 1 \
--update runtime_params.bida_ksize 50
echo "###########################################################"
echo

torchrun --nnodes=1 --nproc_per_node=1 --master_port=$master_port \
tools/eval.py --dist_mode --cfg_file $cfg_file --pretrained_model $pretrained_model \
--eval_data_cfg_file cfgs/common/sequence_datasets/sintel_final.py --eval_batch_size 1 \
--update runtime_params.bida_ksize 50
echo "###########################################################"
echo

torchrun --nnodes=1 --nproc_per_node=1 --master_port=$master_port \
tools/eval.py --dist_mode --cfg_file $cfg_file --pretrained_model $pretrained_model \
--eval_data_cfg_file cfgs/common/sequence_datasets/dynamic_replica.py --eval_batch_size 1 \
--update runtime_params.bida_ksize 20
echo "###########################################################"
echo
