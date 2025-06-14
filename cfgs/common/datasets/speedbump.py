# @Time    : 2024/6/9 18:13
# @Author  : zhangchenming
import os
from stereo.config.lazy import LazyCall
from stereo.datasets.speedbump import SpeedBump
from stereo.datasets.utils import stereo_trans
from stereo.datasets import build_dataloader

from cfgs.common.runtime_params import data_root_dir
from cfgs.common.constants import constants

# data_root_path = os.path.join(data_root_dir, 'CarlaSpeedbumps')
data_root_path = '/file_system/vepfs/public_data/StereoRBHM/CarlaSpeedbumpsV4'

trainv1 = LazyCall(SpeedBump)(
    data_root_path=data_root_path,
    split_file='/baai-cwm-1/baai_cwm_ml/algorithm/xianda.guo/code/Wrl/speedbump/gen_txt/train.txt',
    augmentations=None
)

trainv2 = LazyCall(SpeedBump)(
    data_root_path=data_root_path,
    split_file='/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSeedbumpsPitch20/train1.txt',
    augmentations=None
)

trainv3 = LazyCall(SpeedBump)(
    data_root_path=data_root_path,
    split_file='/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSpeedbumpv3/txt_path/train.txt',
    augmentations=None
)

trainv4 = LazyCall(SpeedBump)(
    data_root_path=data_root_path,
    split_file='/file_system/vepfs/public_data/StereoRBHM/path_txt/v4/train.txt',
    augmentations=None
)

valv1 = LazyCall(SpeedBump)(
    data_root_path=data_root_path,
    split_file='/baai-cwm-1/baai_cwm_ml/algorithm/xianda.guo/code/Wrl/speedbump/gen_txt/val.txt',
    augmentations=[
        LazyCall(stereo_trans.ConstantPad)(target_size=[544, 960]),
        LazyCall(stereo_trans.NormalizeImage)(mean=constants.imagenet_rgb_mean, std=constants.imagenet_rgb_std)
    ]
)

valv2 = LazyCall(SpeedBump)(
    data_root_path=data_root_path,
    split_file='/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSeedbumpsPitch20/val1.txt',
    augmentations=[
        LazyCall(stereo_trans.ConstantPad)(target_size=[544, 960]),
        LazyCall(stereo_trans.NormalizeImage)(mean=constants.imagenet_rgb_mean, std=constants.imagenet_rgb_std)
    ]
)

valv3 = LazyCall(SpeedBump)(
    data_root_path=data_root_path,
    split_file='/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSpeedbumpv3/txt_path/val.txt',
    augmentations=[
        LazyCall(stereo_trans.ConstantPad)(target_size=[544, 960]),
        LazyCall(stereo_trans.NormalizeImage)(mean=constants.imagenet_rgb_mean, std=constants.imagenet_rgb_std)
    ]
)

valv3_bisenet = LazyCall(SpeedBump)(
    data_root_path=data_root_path,
    split_file='/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSpeedbumpv3/txt_path/val_bisenet.txt',
    augmentations=[
        LazyCall(stereo_trans.ConstantPad)(target_size=[544, 960]),
        LazyCall(stereo_trans.NormalizeImage)(mean=constants.imagenet_rgb_mean, std=constants.imagenet_rgb_std)
    ]
)

valv4 = LazyCall(SpeedBump)(
    data_root_path=data_root_path,
    split_file='/file_system/vepfs/public_data/StereoRBHM/path_txt/v4/val.txt',
    augmentations=[
        LazyCall(stereo_trans.ConstantPad)(target_size=[544, 960]),
        LazyCall(stereo_trans.NormalizeImage)(mean=constants.imagenet_rgb_mean, std=constants.imagenet_rgb_std)
    ]
)

valv4_bisenet = LazyCall(SpeedBump)(
    data_root_path=data_root_path,
    split_file='/file_system/vepfs/public_data/StereoRBHM/path_txt/v4/val_bisenet.txt',
    augmentations=[
        LazyCall(stereo_trans.ConstantPad)(target_size=[544, 960]),
        LazyCall(stereo_trans.NormalizeImage)(mean=constants.imagenet_rgb_mean, std=constants.imagenet_rgb_std)
    ]
)

val_loader = LazyCall(build_dataloader)(
    all_dataset=[valv4_bisenet],
    batch_size=1,
    shuffle=False,
    workers=8,
    pin_memory=True)
