# @Time    : 2024/6/9 18:13
# @Author  : zhangchenming
import os
from stereo.config.lazy import LazyCall
from stereo.sequence_datasets.sceneflow_dataset import SequenceSceneFlowDataset
from stereo.sequence_datasets.utils import stereo_trans
from stereo.datasets import build_dataloader

from cfgs.common.runtime_params import data_root_dir
from cfgs.common.constants import constants

data_root_path = os.path.join(data_root_dir, 'SceneFlow')

train_clean = LazyCall(SequenceSceneFlowDataset)(
    data_root_path=data_root_path,
    augmentations=None,
    logger=None,
    dataset_type='frames_cleanpass',
    sample_len=5
)

train_final = LazyCall(SequenceSceneFlowDataset)(
    data_root_path=data_root_path,
    augmentations=None,
    logger=None,
    dataset_type='frames_finalpass',
    sample_len=5
)
