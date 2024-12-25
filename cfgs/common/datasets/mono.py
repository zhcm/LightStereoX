# @Time    : 2024/6/9 18:13
# @Author  : zhangchenming
import os
from stereo.config.lazy import LazyCall
from stereo.datasets.mono_dataset import MonoDataset

from cfgs.common.runtime_params import data_root_dir

data_root_path = os.path.join(data_root_dir, 'depthAnythingData')

train_gl = LazyCall(MonoDataset)(
    data_root_path=data_root_path,
    split_file='/mnt/nas/algorithm/chenming.zhang/misc/Mono/DepthAnythingV2_google_landmarks.txt',
    augmentations=None,
)  # 4976922

train_bdd = LazyCall(MonoDataset)(
    data_root_path=data_root_path,
    split_file='/mnt/nas/algorithm/chenming.zhang/misc/Mono/DepthAnythingV2_bdd100k.txt',
    augmentations=None,
)  # 100000

train_21k = LazyCall(MonoDataset)(
    data_root_path=data_root_path,
    split_file='/mnt/nas/algorithm/chenming.zhang/misc/Mono/DepthAnythingV2_imagenet21K.txt',
    augmentations=None,
)  # 13145689

train_365 = LazyCall(MonoDataset)(
    data_root_path=data_root_path,
    split_file='/mnt/nas/algorithm/chenming.zhang/misc/Mono/DepthAnythingV2_places365.txt',
    augmentations=None,
)  # 2168460

train_lsun = LazyCall(MonoDataset)(
    data_root_path=data_root_path,
    split_file='/mnt/nas/algorithm/chenming.zhang/misc/Mono/DepthAnythingV2_LSUN.txt',
    augmentations=None,
)  # 9898281

train_sa1b = LazyCall(MonoDataset)(
    data_root_path=data_root_path,
    split_file='/baai-cwm-1/baai_cwm_ml/algorithm/xianda.guo/code/ndj/data/DepthAnythingV2/sa-1b.txt',
    augmentations=None,
)  # 9898281

# (1080, 1920, 3)
