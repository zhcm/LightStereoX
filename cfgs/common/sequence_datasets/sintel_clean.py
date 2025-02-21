# @Time    : 2024/6/9 18:13
# @Author  : zhangchenming
import os
from stereo.config.lazy import LazyCall
from stereo.sequence_datasets.sintel_dataset import SequenceSintelDataset
from stereo.sequence_datasets.utils import stereo_trans
from stereo.datasets import build_dataloader

from cfgs.common.runtime_params import data_root_dir
from cfgs.common.constants import constants

data_root_path = os.path.join(data_root_dir, 'Sintel')

train_clean = LazyCall(SequenceSintelDataset)(
    data_root_path=data_root_path,
    augmentations=[LazyCall(stereo_trans.NormalizeImage)(mean=constants.standard_rgb_mean, std=constants.standard_rgb_std)],
    logger=None,
    dataset_type='clean'
)

val_loader = LazyCall(build_dataloader)(
    is_dist=False,
    all_dataset=[train_clean],
    batch_size=1,
    shuffle=False,
    workers=8,
    pin_memory=True)
