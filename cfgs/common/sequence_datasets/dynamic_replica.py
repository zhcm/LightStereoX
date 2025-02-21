# @Time    : 2024/6/9 18:13
# @Author  : zhangchenming
import os
from stereo.config.lazy import LazyCall
from stereo.sequence_datasets.dynamic_replica import DynamicReplicaDataset
from stereo.sequence_datasets.utils import stereo_trans
from stereo.datasets import build_dataloader

from cfgs.common.runtime_params import data_root_dir
from cfgs.common.constants import constants

data_root_path = os.path.join(data_root_dir, 'dynamic_stereo')

train = LazyCall(DynamicReplicaDataset)(
    data_root_path=data_root_path,
    augmentations=None,
    logger=None,
    sample_len=5,
    split="train"
)

val = LazyCall(DynamicReplicaDataset)(
    data_root_path=data_root_path,
    augmentations=None,
    logger=None,
    sample_len=150,
    split="test"
)

val_loader = LazyCall(build_dataloader)(
    is_dist=False,
    all_dataset=[val],
    batch_size=1,
    shuffle=False,
    workers=8,
    pin_memory=True)
