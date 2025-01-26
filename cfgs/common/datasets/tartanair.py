# @Time    : 2024/10/10 22:36
# @Author  : zhangchenming
import os
from stereo.config.lazy import LazyCall
from stereo.datasets.tartanair_dataset import TartanAirDataset
from stereo.datasets.utils import stereo_trans
from stereo.datasets import build_dataloader

from cfgs.common.runtime_params import data_root_dir
from cfgs.common.constants import constants

data_root_path = os.path.join(data_root_dir, 'tartanair')

train = LazyCall(TartanAirDataset)(
    data_root_path=data_root_path,
    split_file='./data/TartanAir/all_306637.txt',
    augmentations=None
)  # 306637

# (480, 640)
