# @Time    : 2024/6/22 00:56
# @Author  : zhangchenming
from stereo.config.lazy import LazyCall
from stereo.datasets.middlebury_dataset import MiddleburyDataset
from stereo.datasets.utils import stereo_trans
from stereo.datasets import build_dataloader

from cfgs.common.constants import constants

data_root_path = '/mnt/nas/algorithm/chenming.zhang/dataset/Middlebury'

trainval = LazyCall(MiddleburyDataset)(
    data_root_path=data_root_path,
    split_file='./data/Middlebury/middeval3_train_h.txt',
    augmentations=[
        LazyCall(stereo_trans.DivisiblePad)(divisor=32, mode='tr'),
        LazyCall(stereo_trans.NormalizeImage)(mean=constants.imagenet_rgb_mean, std=constants.imagenet_rgb_std)
    ],
)

# 泛化测试，使用所有训练数据进行eval
val_loader = LazyCall(build_dataloader)(
    all_dataset=[trainval],
    batch_size=1,
    workers=8,
    pin_memory=True,
    shuffle=False)
