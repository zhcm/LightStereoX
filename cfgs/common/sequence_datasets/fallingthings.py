# @Time    : 2024/6/9 18:13
# @Author  : zhangchenming
import os
from stereo.config.lazy import LazyCall
from stereo.sequence_datasets.fallingthings import FallingThings

from cfgs.common.runtime_params import data_root_dir

data_root_path = os.path.join(data_root_dir, 'FallingThings')

train = LazyCall(FallingThings)(
    data_root_path=data_root_path,
    augmentations=None,
    logger=None,
    sample_len=5
)
