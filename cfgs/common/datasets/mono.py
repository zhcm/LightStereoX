# @Time    : 2024/6/9 18:13
# @Author  : zhangchenming
import os
from stereo.config.lazy import LazyCall
from stereo.datasets.mono_dataset import MonoDataset
from stereo.datasets.utils import stereo_trans
from stereo.datasets import build_dataloader

from cfgs.common.runtime_params import data_root_dir
from cfgs.common.constants import constants

data_root_path = os.path.join(data_root_dir, 'depthAnythingData')

train_gl = LazyCall(MonoDataset)(
    data_root_path=data_root_path,
    split_file='./data/Mono/DepthAnythingV2_google_landmarks.txt',
    augmentations=None,
)

train_bdd = LazyCall(MonoDataset)(
    data_root_path=data_root_path,
    split_file='./data/Mono/DepthAnythingV2_bdd100k.txt',
    augmentations=None,
)

train_21k = LazyCall(MonoDataset)(
    data_root_path=data_root_path,
    split_file='./data/Mono/DepthAnythingV2_imagenet21K.txt',
    augmentations=None,
)

train_365 = LazyCall(MonoDataset)(
    data_root_path=data_root_path,
    split_file='./data/Mono/DepthAnythingV2_places365.txt',
    augmentations=None,
)

# (1080, 1920, 3)
