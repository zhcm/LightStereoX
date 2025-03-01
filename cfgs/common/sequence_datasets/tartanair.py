# @Time    : 2024/10/10 22:36
# @Author  : zhangchenming
import os
from stereo.config.lazy import LazyCall
from stereo.sequence_datasets.tartanair import TartanAir

from cfgs.common.runtime_params import data_root_dir

data_root_path = os.path.join(data_root_dir, 'tartanair')

train = LazyCall(TartanAir)(
    data_root_path=data_root_path,
    augmentations=None,
    logger=None,
    sample_len=5
)
