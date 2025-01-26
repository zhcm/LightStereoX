# @Time    : 2024/6/9 18:13
# @Author  : zhangchenming
import os
from stereo.config.lazy import LazyCall
from stereo.datasets.spring_dataset import SpringDataset, SpringTestDataset
from stereo.datasets.utils import stereo_trans
from stereo.datasets import build_dataloader

from cfgs.common.runtime_params import data_root_dir
from cfgs.common.constants import constants

data_root_path = os.path.join(data_root_dir, 'spring')

train = LazyCall(SpringDataset)(
    data_root_path=data_root_path,
    split_file='./data/Spring/spring_train.txt',
    augmentations=None,
    return_right_disp=True
)  # 5000

test = LazyCall(SpringTestDataset)(
    data_root_path=data_root_path,
    split_file='./data/Spring/spring_test.txt',
    augmentations=[
        LazyCall(stereo_trans.ConstantPad)(target_size=[1088, 1920]),
        LazyCall(stereo_trans.NormalizeImage)(mean=constants.imagenet_rgb_mean, std=constants.imagenet_rgb_std)
    ],
)

test_loader = LazyCall(build_dataloader)(
    all_dataset=[test],
    batch_size=1,
    shuffle=False,
    workers=8,
    pin_memory=True)
