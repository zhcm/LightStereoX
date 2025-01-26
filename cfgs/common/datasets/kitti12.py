# @Time    : 2024/6/9 18:13
# @Author  : zhangchenming
import os
from stereo.config.lazy import LazyCall
from stereo.datasets.kitti_dataset import KittiDataset, KittiTestDataset
from stereo.datasets.utils import stereo_trans
from stereo.datasets import build_dataloader

from cfgs.common.runtime_params import data_root_dir
from cfgs.common.constants import constants

data_root_path = os.path.join(data_root_dir, 'kitti12')

train = LazyCall(KittiDataset)(
    data_root_path=data_root_path,
    split_file='./data/KITTI2012/kitti12_train180.txt',
    augmentations=None,
    return_right_disp=True,
    use_noc=False
)

val = LazyCall(KittiDataset)(
    data_root_path=data_root_path,
    split_file='./data/KITTI2012/kitti12_val14.txt',
    augmentations=[
        LazyCall(stereo_trans.ConstantPad)(target_size=[384, 1248], mode='tr'),
        LazyCall(stereo_trans.NormalizeImage)(mean=constants.imagenet_rgb_mean, std=constants.imagenet_rgb_std)
    ],
    return_right_disp=True,
    use_noc=False
)

trainval = LazyCall(KittiDataset)(
    data_root_path=data_root_path,
    split_file='./data/KITTI2012/kitti12_train194.txt',
    augmentations=[
        LazyCall(stereo_trans.ConstantPad)(target_size=[384, 1248], mode='tr'),
        LazyCall(stereo_trans.NormalizeImage)(mean=constants.imagenet_rgb_mean, std=constants.imagenet_rgb_std)
    ],
    return_right_disp=True,
    use_noc=False
)

# 泛化测试，使用所有训练数据进行eval
val_loader = LazyCall(build_dataloader)(
    all_dataset=[trainval],
    batch_size=1,
    workers=8,
    pin_memory=True,
    shuffle=False)

test = LazyCall(KittiTestDataset)(
    data_root_path=data_root_path,
    split_file='./data/KITTI2012/kitti12_test.txt',
    augmentations=[
        LazyCall(stereo_trans.ConstantPad)(target_size=[384, 1248], mode='tr'),
        LazyCall(stereo_trans.NormalizeImage)(mean=constants.imagenet_rgb_mean, std=constants.imagenet_rgb_std)
    ]
)

test_loader = LazyCall(build_dataloader)(
    all_dataset=[test],
    batch_size=1,
    workers=8,
    pin_memory=True,
    shuffle=False)

# trainval (374, 1238, 3), (375, 1242, 3), (370, 1226, 3), (376, 1241, 3)
