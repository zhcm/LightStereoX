# @Time    : 2024/6/9 18:13
# @Author  : zhangchenming
import os
from stereo.config.lazy import LazyCall
from stereo.sequence_datasets.kitti_odometry_2012 import SequenceKittiDataset

from cfgs.common.runtime_params import data_root_dir

data_root_path = os.path.join('/baai-cwm-nas/public_data/scenes/videodepthdata/kitti_odometry_2012')

train = LazyCall(SequenceKittiDataset)(
    data_root_path=data_root_path,
    augmentations=None,
    logger=None,
    sample_len=5
)
