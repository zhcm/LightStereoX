# @Time    : 2024/6/9 18:13
# @Author  : zhangchenming
import os
from stereo.config.lazy import LazyCall
from stereo.datasets.fallingthings_dataset import FallingThingsDataset

from cfgs.common.runtime_params import data_root_dir

data_root_path = os.path.join(data_root_dir, 'FallingThings')

train = LazyCall(FallingThingsDataset)(
    data_root_path=data_root_path,
    split_file='./data/FallingThings/fat_train.txt',
    augmentations=None,
    return_right_disp=True
)
