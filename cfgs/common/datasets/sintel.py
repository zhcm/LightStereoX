# @Time    : 2024/6/9 18:13
# @Author  : zhangchenming
import os
from stereo.config.lazy import LazyCall
from stereo.datasets.sintel_dataset import SintelDataset
from stereo.datasets.utils import stereo_trans
from stereo.datasets import build_dataloader

from cfgs.common.runtime_params import data_root_dir
from cfgs.common.constants import constants

data_root_path = os.path.join(data_root_dir, 'Sintel')

train = LazyCall(SintelDataset)(
    data_root_path=data_root_path,
    split_file='./data/Sintel/Sintel_final_all.txt',
    augmentations=None
)  # 1064

# (436, 1024, 3)
