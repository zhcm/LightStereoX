# @Time    : 2024/6/9 18:13
# @Author  : zhangchenming
import os
from stereo.config.lazy import LazyCall
from stereo.datasets.sceneflow_dataset import SceneFlowDataset
from stereo.datasets.utils import stereo_trans
from stereo.datasets import build_dataloader

from cfgs.common.runtime_params import data_root_dir
from cfgs.common.constants import constants

data_root_path = os.path.join(data_root_dir, 'SceneFlow')

train = LazyCall(SceneFlowDataset)(
    data_root_path=data_root_path,
    split_file='./data/SceneFlow/sceneflow_finalpass_train_35454.txt',
    augmentations=None,
    return_right_disp=True
)  # 35454

val = LazyCall(SceneFlowDataset)(
    data_root_path=data_root_path,
    split_file='./data/SceneFlow/sceneflow_finalpass_test_4370.txt',
    augmentations=[
        LazyCall(stereo_trans.ConstantPad)(target_size=[544, 960]),
        LazyCall(stereo_trans.NormalizeImage)(mean=constants.imagenet_rgb_mean, std=constants.imagenet_rgb_std)
    ],
    return_right_disp=True
)  # 4370

val_loader = LazyCall(build_dataloader)(
    all_dataset=[val],
    batch_size=2,
    shuffle=False,
    workers=8,
    pin_memory=True)

# (540, 960, 3)
