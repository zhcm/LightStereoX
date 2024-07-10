# @Time    : 2024/6/9 18:13
# @Author  : zhangchenming
import os
from stereo.config.lazy import LazyCall
from stereo.datasets.sceneflow_dataset import FlyingThings3DSubsetDataset
from stereo.datasets.utils import stereo_trans
from stereo.datasets import build_dataloader

from cfgs.common.runtime_params import data_root_dir
from cfgs.common.constants import constants

data_root_path = os.path.join(data_root_dir, 'SceneFlow')

train = LazyCall(FlyingThings3DSubsetDataset)(
    data_root_path=data_root_path,
    split_file='./data/SceneFlow/flyingthings3d_sttr_train.txt',
    augmentations=None,
    return_occ_mask=True,
    zeroing_occ=True
)

val = LazyCall(FlyingThings3DSubsetDataset)(
    data_root_path=data_root_path,
    split_file='./data/SceneFlow/flyingthings3d_sttr_test.txt',
    augmentations=[
        LazyCall(stereo_trans.ConstantPad)(target_size=[540, 960]),
        LazyCall(stereo_trans.NormalizeImage)(mean=constants.imagenet_rgb_mean, std=constants.imagenet_rgb_std)
    ],
    return_occ_mask=True,
    zeroing_occ=True
)

val_loader = LazyCall(build_dataloader)(
    all_dataset=[val],
    batch_size=1,
    shuffle=False,
    workers=8,
    pin_memory=True)
