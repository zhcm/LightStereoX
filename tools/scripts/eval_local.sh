set -x
export CUDA_VISIBLE_DEVICES=1,2,4,5,6,7

cfg_file="cfgs/nmrf/nmrf_swint_sceneflow.py"
pretrained_model="output/MixDataset/NMRF/mix7gl/ckpt/epoch_0/pytorch_model.bin"
nproc_per_node=6
master_port=2335

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node --master_port=$master_port \
tools/eval.py --dist_mode --cfg_file $cfg_file --pretrained_model $pretrained_model \
--eval_data_cfg_file cfgs/common/datasets/kitti12.py --eval_batch_size 1

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node --master_port=$master_port \
tools/eval.py --dist_mode --cfg_file $cfg_file --pretrained_model $pretrained_model \
--eval_data_cfg_file cfgs/common/datasets/kitti15.py --eval_batch_size 1

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node --master_port=$master_port \
tools/eval.py --dist_mode --cfg_file $cfg_file --pretrained_model $pretrained_model \
--eval_data_cfg_file cfgs/common/datasets/middlebury.py --eval_batch_size 1

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node --master_port=$master_port \
tools/eval.py --dist_mode --cfg_file $cfg_file --pretrained_model $pretrained_model \
--eval_data_cfg_file cfgs/common/datasets/eth3d.py --eval_batch_size 1

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node --master_port=$master_port \
tools/eval.py --dist_mode --cfg_file $cfg_file --pretrained_model $pretrained_model \
--eval_data_cfg_file cfgs/common/datasets/drivingstereo.py --eval_batch_size 1
